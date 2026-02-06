# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import math

import torch
import torch.nn.functional as F

from dynamic_tokenization.models.generate import generate, generate_dynamic, generate_llamagen
from dynamic_tokenization.models.gpt import GPT_models
from dynamic_tokenization.models.dynamic_gpt import DGPT_models
from dynamic_tokenization.models.patcher import PatcherArgs
from dynamic_tokenization.tokenizer_image.vq_model import VQ_models
from dynamic_tokenization.utils.conf import setup_default_config_values
from dynamic_tokenization.utils.logger import create_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch._dynamo.config.optimize_ddp = False
import argparse
import os

import numpy as np
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin, set_seed, tqdm
from omegaconf import OmegaConf
from PIL import Image

from safetensors.torch import load_file

def main(experiment_dir, config, args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    #################### Accelerator and Checkpoint setup ####################
    # dynamo_plugin = TorchDynamoPlugin(
    #     backend="inductor",  # type: ignore
    #     mode=config.accelerator.inductor_mode,  # Options: "default", "reduce-overhead", "max-autotune"
    #     dynamic=True if "DGPT" in config.model.gpt_model else False
    # )
    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
        # dynamo_plugin=dynamo_plugin,
    )
    device = accelerator.device
    set_seed(args.global_seed, device_specific=True)

    # setup logging
    logger = create_logger(rank=accelerator.process_index)

    # global vars
    if "DGPT" in config.model.gpt_model:
        logger.info("Dynamic GPT sampling.....")
        MODEL_DICT = DGPT_models
        GENERATE_FN = generate_dynamic
        generate_fn_kwargs = {}
        model_kwargs = {
            "predict_eoi_token": config.model.predict_eoi_token, 
            "embedding_type": config.model.embedding_type,
            "encoder_block_causal": config.model.encoder_block_causal,
            "use_ca_rope": config.training.use_ca_rope,
        }
    else:
        logger.info("Llama GPT sampling.....")
        MODEL_DICT = GPT_models
        GENERATE_FN = generate_llamagen
        generate_fn_kwargs = {}
        model_kwargs = {}

    #################### Model ####################
    # setup model
    if config.model.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = config.model.dropout_p
    latent_size = config.dataset.image_size // config.model.downsample_size
    model = MODEL_DICT[config.model.gpt_model](
        vocab_size=config.model.codebook_size,
        block_size=latent_size**2,
        num_classes=config.dataset.num_classes,
        cls_token_num=config.model.cls_token_num,
        model_type=config.model.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=config.model.drop_path_rate,
        token_dropout_p=config.model.token_dropout_p,
        **model_kwargs
    ).to(device)
    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # setup tokenizer
    vq_model = VQ_models[config.model.vq_model](
        codebook_size=config.model.codebook_size,
        codebook_embed_dim=config.model.codebook_embed_dim,
    )
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(config.model.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint
    logger.info(f"Image tokenizer is loaded")

    # setup patcher and prepare
    patcher = None
    model.move_buffers_to_device(device)
    
    ################## Resume Training ##################
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    assert os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("epoch")]
    saved_ckpt_dirs = sorted(saved_ckpt_dirs)
    if args.epoch is None:
        epoch = 'final'
        ckpt_dir = f"{checkpoint_dir}/final/"
    else:
        epoch = args.epoch
        ckpt_dir = f"{checkpoint_dir}/epoch_{epoch:08d}/"
    logger.info(f"Resuming from {ckpt_dir}...")
    # check if safetensors or .pt file
    if os.path.exists(f"{ckpt_dir}/model.safetensors"):
        model_state_dict = load_file(f"{ckpt_dir}/model.safetensors")
    elif os.path.exists(f"{ckpt_dir}/model.pt"):
        model_state_dict = torch.load(f"{ckpt_dir}/model.pt", map_location="cpu")
        model_state_dict = model_state_dict["model"]
    else:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    model.load_state_dict(model_state_dict)
    
    # prepare model
    if "DGPT" in config.model.gpt_model:
        patcher = PatcherArgs(**config.patcher).build()
        patcher.eval()
        for param in patcher.parameters():
            param.requires_grad = False
        patcher = patcher.to(device)
        patcher.move_buffers_to_device(device)
        logger.info(patcher.patcher_args)
        model, patcher = accelerator.prepare(model, patcher)
        generate_fn_kwargs["patcher"] = patcher
    else:
        model = accelerator.prepare(model)        

    # NOTE: This needs to be done after accelerator.load_state since it loads random generator as well
    set_seed(args.global_seed, device_specific=True)
    
    # setup checkpoint path
    if args.output_dir:
        samples_dir = os.path.join(experiment_dir, args.output_dir)
    else:
        samples_dir = os.path.join(experiment_dir, "samples")
    folder_name = f"epoch-{epoch}-topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-cfg-{args.cfg_scale}-seed-{args.global_seed}"
    samples_dir = f"{samples_dir}/{folder_name}"
    if accelerator.is_main_process:
        os.makedirs(samples_dir, exist_ok=True)
    accelerator.wait_for_everyone()
    logger.info(f"Samples directory: {samples_dir}")

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * accelerator.num_processes
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    logger.info(f"Total number of images that will be sampled: {total_samples}")

    assert total_samples % accelerator.num_processes == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // accelerator.num_processes)
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    total = 0
    model.eval()
    running_patch_length = 0.0
    for iteration in tqdm(range(iterations)):
        # Sample inputs:
        c_indices = torch.randint(0, config.dataset.num_classes, (n,), device=device)
        qzshape = [len(c_indices), config.model.codebook_embed_dim, latent_size, latent_size]

        # sample
        index_sample, mean_patch_len = GENERATE_FN(
            model,
            c_indices,
            latent_size**2,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
            **generate_fn_kwargs
        )
        running_patch_length = running_patch_length * 0.99 + mean_patch_len * 0.01

        # decode
        samples = vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
        if args.image_size_eval != config.dataset.image_size:
            samples = F.interpolate(samples, size=(args.image_size_eval, args.image_size_eval), mode="bicubic")
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * accelerator.num_processes + accelerator.process_index + total
            Image.fromarray(sample).save(f"{samples_dir}/{index:06d}.png")
        total += global_batch_size

        # sync
        accelerator.wait_for_everyone()
        
        if iteration % 10 == 0:
            logger.info(f"Iteration {iteration}, average patch length at inference: {running_patch_length:.4f}")

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--image-size-eval", type=int, choices=[256, 384, 512], default=256)
    parser.add_argument("--cfg-scale", type=float, default=1.75)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--per-proc-batch-size", type=int, default=24)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=0, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    
    # patcher arguments
    parser.add_argument("--patcher.entropy_model_checkpoint_config", type=str)
    parser.add_argument("--patcher.entropy_model_checkpoint", type=str)
    parser.add_argument("--patcher.threshold", type=float)
    parser.add_argument("--patcher.max_patch_length", type=int)
    parser.add_argument("--patcher.patch_size", type=float)
    
    args = parser.parse_args()
    args_dict = {k: v for k, v in vars(args).items() if v is not None}
    cli_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in args_dict.items()])

    # load omegaconfig
    config = OmegaConf.load(f"{args.experiment_dir}/config.yaml")
    
    # merge with script args
    config = OmegaConf.merge(config, cli_cfg)

    # check if an argument is absent, if so add default
    config = setup_default_config_values(config)

    main(args.experiment_dir, config, args)
