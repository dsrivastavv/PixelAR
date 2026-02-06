# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch

from dynamic_tokenization.models.generate import generate, generate_dynamic, generate_llamagen
from dynamic_tokenization.models.gpt import GPT_models
from dynamic_tokenization.models.dynamic_gpt import DGPT_models
from dynamic_tokenization.models.patcher import PatcherArgs
from dynamic_tokenization.tokenizer_image.vq_model import VQ_models
from dynamic_tokenization.utils.conf import setup_default_config_values
from dynamic_tokenization.utils.logger import create_logger
from torchvision.utils import save_image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
import time

from accelerate import Accelerator
from accelerate.utils import set_seed
from omegaconf import OmegaConf

from safetensors.torch import load_file

CFG_SCALES = [1.75, 1.9, 2.0, 2.1]

def main(experiment_dir, config, args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    #################### Accelerator and Checkpoint setup ####################
    accelerator = Accelerator(
        mixed_precision=config.accelerator.mixed_precision,
    )
    device = accelerator.device

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

    # setup patcher
    patcher = None
    if "DGPT" in config.model.gpt_model:
        patcher = PatcherArgs(**config.patcher).build()
        patcher.eval()
        for param in patcher.parameters():
            param.requires_grad = False
        patcher = patcher.to(device)
        patcher.move_buffers_to_device(device)
        logger.info(patcher.patcher_args)
        generate_fn_kwargs["patcher"] = patcher
        
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

    # prepare with accelerate
    model.move_buffers_to_device(device)
    model = accelerator.prepare(model)

    # setup seed
    # NOTE: This needs to be done after accelerator.load_state since it loads random generator as well
    set_seed(args.global_seed, device_specific=True)

    # setup checkpoint path
    model_type = config.model.gpt_model.split('-')[1]
    model_res = config.dataset.image_size
    for cfg_scale in CFG_SCALES:
        print(f"Generating samples with CFG scale: {cfg_scale}")
        file_name = f"model-{model_type}-{model_res}-epoch-{epoch}-cfg-{cfg_scale}-topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-seed-{args.global_seed}.jpg"
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            file_path = f"{args.output_dir}/{file_name}"
        else:
            file_path = f"{experiment_dir}/{file_name}"
            
        generated_imgs = []
        for class_id in args.class_ids:
            print(f"Generating images for class id: {class_id}")
            for _ in range(8 // args.batch_size):
                # Labels to condition the model with (feel free to change):
                class_labels = [class_id] * args.batch_size
                c_indices = torch.tensor(class_labels, device=device)
                qzshape = [len(class_labels), config.model.codebook_embed_dim, latent_size, latent_size]

                t1 = time.time()
                model.eval()
                index_sample, _ = GENERATE_FN(
                    model,
                    c_indices,
                    latent_size**2,
                    cfg_scale=cfg_scale,
                    cfg_interval=args.cfg_interval,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    sample_logits=True,
                    **generate_fn_kwargs
                )
                sampling_time = time.time() - t1
                logger.info(f"gpt sampling takes about {sampling_time:.2f} seconds.")

                t2 = time.time()
                torch.cuda.empty_cache()
                
                samples = vq_model.decode_code(index_sample, qzshape)  # output value is between [-1, 1]
                decoder_time = time.time() - t2
                logger.info(f"decoder takes about {decoder_time:.2f} seconds.")
                
                generated_imgs.append(samples.detach().cpu())

        # Save and display images:
        generated_imgs = torch.cat(generated_imgs, dim=0)
        save_image(generated_imgs, file_path, nrow=8, normalize=True, value_range=(-1, 1))
        print(f"image is saved to {file_path}")

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="visualizations")
    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=2000, help="top-k value to sample with")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature value to sample with")
    parser.add_argument("--top-p", type=float, default=1.0, help="top-p value to sample with")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size for sampling")
    
    # class-id
    # 33: loggerhead sea turtle
    # 88: macaw
    # 89: sulphur-crested cockatoo
    # 207: golden retriever
    # 250: husky
    # 270: arctic wolf
    # 279: arctic fox
    # 291: lion
    # 360: otter
    # 387: red panda
    # 388: panda
    # 417: balloon
    # 537: dog sled
    # 812: space shuttle
    # 928: ice cream
    # 972: cliff drop-off
    # 973: coral reef
    # 974: geyser
    # 975: lake shore
    # 979: seashore
    # 980: volcano
    
    # parser.add_argument("--class-ids", type=int, nargs='+', default=[33, 88, 89, 199, 207, 250, 270, 279, 291, 357, 360, 387, 388, 417, 537, 812, 928, 972, 973, 974, 975, 979, 980])
    parser.add_argument("--class-ids", type=int, nargs='+', default=[33, 88, 89, 207, 250, 270, 291, 360, 388, 974])
    
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
