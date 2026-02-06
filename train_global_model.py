# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
from functools import partial
from pathlib import Path
import torch
import gc

from PixelAR.dataset.imagenet import CustomDataset
from PixelAR.dataset.patch_dataset import PatchDataset, collate_fn as patch_collate_fn
from PixelAR.evaluation.evaluate import evaluate_reconstruction
from PixelAR.models.dynamic_gpt import DGPT_models
from PixelAR.models.patcher import PatcherArgs
from PixelAR.models.utils import save_with_retries
from PixelAR.tokenizer_image.vq_model import VQ_models
from PixelAR.utils.conf import setup_default_config_values
from PixelAR.utils.gpu_monitor import GPUMemoryMonitor
from PixelAR.utils.logger import create_logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import inspect
import os
import time
from datetime import datetime

import wandb
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration, TorchDynamoPlugin, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


def build_imagenet_code(code_path, image_size, preprocessed_entropy_dir=None):
    feature_dir = f"{code_path}/imagenet{image_size}_codes"
    label_dir = f"{code_path}/imagenet{image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(
        label_dir
    ), f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir, preprocessed_entropy_dir=preprocessed_entropy_dir)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def create_optimizer(model, weight_decay, learning_rate, beta1, beta2, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        # weight_decay=weight_decay,
        betas=(beta1, beta2),
        **extra_args,
    )
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

# @torch.no_grad
# def check_nan_in_loss_and_grads(loss, model):
#     # Check for NaN in loss
#     is_nan = False
#     if torch.isnan(loss).any():
#         print("❌ NaN detected in loss!")
#         is_nan = True

#     # Check for NaNs in gradients
#     for name, param in model.named_parameters():
#         if param.grad is not None and torch.isnan(param.grad).any():
#             print(f"❌ NaN detected in gradients of parameter: {name}")
#             is_nan = True
    
#     for name, param in model.named_parameters():
#         if torch.isnan(param).any():
#             print(f"❌ NaN detected in parameter: {name}")
#             is_nan = True

#     return is_nan

# import torch
# import torch.nn as nn

# def detect_nan_hook(module, grad_input, grad_output, module_name, model=None):
#     print(module, module_name)
#     for i, grad in enumerate(grad_input):
#         if grad is not None and torch.isnan(grad).any():
#             print(f"NaN detected in {module.__class__.__name__} (input {i})")
#             # import pdb; pdb.set_trace()  # or raise an error if preferred

#     for i, grad in enumerate(grad_output):
#         if grad is not None and torch.isnan(grad).any():
#             print(f"NaN detected in {module.__class__.__name__} (output {i})")

# def add_hooks(model):
#     model.module.local_decoder.output.register_full_backward_hook(partial(detect_nan_hook, module_name="output", model=model))
#     model.module.local_decoder.norm.register_full_backward_hook(partial(detect_nan_hook, module_name="norm", model=model))
#     model.module.local_decoder.cross_attn_layers[-1].register_full_backward_hook(partial(detect_nan_hook, module_name="CA", model=model))
#     model.module.local_decoder.cross_attn_layers[-1].wq.register_full_backward_hook(partial(detect_nan_hook, module_name="wq", model=model))
#     model.module.local_decoder.cross_attn_layers[-1].wk.register_full_backward_hook(partial(detect_nan_hook, module_name="wk", model=model))






def main(config, uuid=None, debug=False):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    #################### Accelerator and Checkpoint setup ####################
    if not config.resume_dir:
        if uuid:
            timestamp=uuid
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H")
        experiment_dir = os.path.join(config.results_dir, config.exp_name, str(timestamp))
    else:
        experiment_dir = config.resume_dir
    accelerator_config = ProjectConfiguration(project_dir=experiment_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True if debug else False)
    if debug:
        dynamo_plugin = None
    else:
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # type: ignore
            mode=config.accelerator.inductor_mode,  # Options: "default", "reduce-overhead", "max-autotune"
            dynamic=True
        )
    accelerator = Accelerator(
        project_config=accelerator_config,
        kwargs_handlers=[ddp_kwargs],
        mixed_precision=config.accelerator.mixed_precision,
        gradient_accumulation_steps=config.accelerator.gradient_accumulation_steps,
        dynamo_plugin=dynamo_plugin,
        log_with=["tensorboard", config.checkpoint.log_with],
    )
    device = accelerator.device
    set_seed(config.global_seed, device_specific=True)

    # wandb setup
    accelerator.init_trackers(
        project_name="DynamicAR",
        init_kwargs={
            "wandb": {
                "entity": config.training.wandb.wandb_entity,
                "config": dict(config),
                "name": config.exp_name,
                "dir": experiment_dir,
            }
        },
        config=flatten_dict(OmegaConf.to_container(config)),
    )

    # setup checkpoint path
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # dump config
        OmegaConf.save(config=config, f=f"{experiment_dir}/config.yaml")
    accelerator.wait_for_everyone()

    # setup logging
    logger = create_logger(experiment_dir, rank=accelerator.process_index, debug=debug)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(accelerator.state)
    logger.info(config)

    # setup GPU profiling
    gpu_memory_monitor = GPUMemoryMonitor(device)
    logger.info(
        f"GPU capacity: {gpu_memory_monitor.device_name} ({gpu_memory_monitor.device_index}) "
        f"with {gpu_memory_monitor.device_capacity_gib:.2f}GiB memory",
    )
    logger.info(f"GPU memory usage: {gpu_memory_monitor}")

    #################### Data, Model, Optimization ####################
    # setup patcher
    patcher = PatcherArgs(use_preprocessed_entropy=not config.training.run_patcher, **config.patcher).build()
    patcher.eval()
    for param in patcher.parameters():
        param.requires_grad = False
    logger.info(patcher.patcher_args)

    # setup dataset
    dataset = build_imagenet_code(
        config.dataset.code_path, 
        config.dataset.image_size, 
        preprocessed_entropy_dir=f"{Path(config.patcher.entropy_model_checkpoint_config).parent}/processed_entropy" if not config.training.run_patcher else None
    )
    flip_info = "with" if dataset.flip else "without"
    aug_info = 10 if "ten_crop" in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(
        f"Dataset contains {len(dataset):,} images ({config.dataset.code_path}) "
        f"{flip_info} flip augmentation and {aug_info} crop augmentation"
    )
    dataset = PatchDataset(
        dataset,  
        predict_eoi_token=config.model.predict_eoi_token, 
        patcher=patcher,
    )
    if not config.training.run_patcher:
        collate_fn = partial(patch_collate_fn, packed_input=config.training.packed_inputs)
    else:
        collate_fn = None
    per_gpu_batch_size = int(
        config.training.global_batch_size // accelerator.num_processes // config.accelerator.gradient_accumulation_steps
    )
    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.training.num_workers > 0 else False,
        prefetch_factor=8 if config.training.num_workers > 0 else None,
        collate_fn=collate_fn
    )

    # setup model
    if config.model.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = config.model.dropout_p
    latent_size = config.dataset.image_size // config.model.downsample_size
    model = DGPT_models[config.model.gpt_model](
        vocab_size=config.model.codebook_size,
        block_size=latent_size**2,
        num_classes=config.dataset.num_classes,
        cls_token_num=config.model.cls_token_num,
        model_type=config.model.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=config.model.drop_path_rate,
        token_dropout_p=config.model.token_dropout_p,
        predict_eoi_token=config.model.predict_eoi_token,
        embedding_type=config.model.embedding_type,
        encoder_block_causal=config.model.encoder_block_causal,
        packed_inputs=config.training.packed_inputs,
        use_ca_rope=config.training.use_ca_rope,
        gradient_checkpointing=config.training.gradient_checkpointing,
    ).to(device)
    predict_eoi_token = model.config.predict_eoi_token
    EOI_TOKEN = model.EOI_TOKEN
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Predict EOI token: {predict_eoi_token} with EOI_TOKEN: {EOI_TOKEN}")

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

    # setup optimizer
    optimizer = create_optimizer(
        model,
        config.optimizer.weight_decay,
        config.optimizer.lr,
        config.optimizer.beta1,
        config.optimizer.beta2,
        logger,
    )

    # prepare model
    model.move_buffers_to_device(device)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    # prepare patcher
    if config.training.run_patcher:
        logger.info("Preparing patcher....")
        patcher = patcher.to(device)
        patcher.entropy_model.move_buffers_to_device(device)
        patcher = accelerator.prepare(patcher)

    ################## Resume Training ##################
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("epoch")]
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        ckpt_dir = f"{checkpoint_dir}/{saved_ckpt_dirs[-1]}/"
        logger.info(f"Resuming from {ckpt_dir}")
        accelerator.load_state(ckpt_dir)
        start_epoch = int(saved_ckpt_dirs[-1].split("_")[-1])+1
        train_steps = int((start_epoch) * len(data_loader)) // config.accelerator.gradient_accumulation_steps
    else:
        start_epoch = 0
        train_steps = 0

    #################### Training Loop ####################
    num_epochs = config.training.epochs
    running_loss, running_grad_norm, running_patch_length, start_time = 0, 0, 0.0, time.time()
    logger.info(f"Training for {num_epochs} epochs starting at epoch {start_epoch}...")
    accelerator.wait_for_everyone()
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for batch in data_loader:
            # torch.compiler.cudagraph_mark_step_begin()
            model.train()
            tokens, labels = batch[0], batch[1]

            # get patch lengths
            if config.training.run_patcher:
                with torch.no_grad():
                    patch_lens, _ = patcher(
                        tokens[:, :-1], 
                        labels, 
                        include_next_token=True, 
                        include_eoi_token=predict_eoi_token,
                        attn_impl=config.model.attn_impl
                    )
            else:
                patch_lens = batch[2]

            # eoi token prediction
            if predict_eoi_token:
                idx = tokens
                targets = torch.concat([tokens, torch.tensor([EOI_TOKEN], device=device).repeat(tokens.shape[0]).view(-1,1)], dim=-1).contiguous()
            else:
                idx = tokens[:, :-1]
                targets = tokens
            with accelerator.accumulate(model):
                if config.training.packed_inputs:
                    bsz, seq_len = idx.shape
                    patch_lens_w_next, patch_seqlens, _ = batch[3:]
                    token_seqlens = torch.tensor([seq_len], device=device, dtype=torch.long).repeat(bsz)
                    logits, loss = model(
                        idx=idx.contiguous().view(-1), 
                        cond_idx=labels,
                        patch_lens=patch_lens,
                        targets=targets.contiguous().view(-1),
                        patch_lens_w_next=patch_lens_w_next,
                        token_seqlens=token_seqlens,
                        patch_seqlens=patch_seqlens,
                    )
                    logits = logits.reshape(bsz, seq_len+1, -1)
                else:
                    logits, loss = model(
                        idx=idx, 
                        cond_idx=labels,
                        patch_lens=patch_lens,
                        targets=targets,
                        attn_impl=config.model.attn_impl
                    )

                # backward pass
                accelerator.backward(loss)

                # gradient clipping
                if accelerator.sync_gradients and config.optimizer.max_grad_norm != 0.0:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)
                    running_grad_norm += grad_norm.item()

                # updates
                optimizer.step()
                optimizer.zero_grad()

                # running loss calculation
                loss = loss.detach()
                logits = logits[:, :-1].detach() if predict_eoi_token else logits.detach()
                running_loss += loss.item() / config.accelerator.gradient_accumulation_steps

            if accelerator.sync_gradients:
                train_steps += 1
                model.eval()

                # log metrics
                if train_steps % config.checkpoint.log_every == 0:
                    # loss and grad norm
                    local_loss = torch.tensor(running_loss / config.checkpoint.log_every, device=device)
                    average_loss = accelerator.gather(local_loss).mean().item()
                    average_grad_norm = running_grad_norm / config.checkpoint.log_every

                    # running average patch length
                    running_patch_length = running_patch_length * 0.99 + (patch_lens.sum() / (patch_lens > 0).sum()) * 0.01


                    # gpu monitor
                    gpu_mem_stats = gpu_memory_monitor.get_peak_stats(logger=logger)

                    # speed
                    end_time = time.time()
                    average_time = (end_time - start_time) / config.checkpoint.log_every
                    start_time = time.time()

                    # log info
                    logger.info(
                        f"Epoch {epoch:08d} | Step {train_steps:08d} | Loss {average_loss:.4f} | Time {average_time:.4f}s | Grad Norm {average_grad_norm:.4f} | LR {optimizer.param_groups[0]['lr']:.5f} |  patch length: {running_patch_length} | mem: {gpu_mem_stats.max_reserved_gib:.0f}% | pow: {gpu_mem_stats.power_draw/1000} W"
                    )
                    logger_dict = {
                        "train/loss": average_loss,
                        "train/avg_patch_length": running_patch_length,
                        "benchmark/memory": gpu_mem_stats._asdict(),
                        "benchmark/time": average_time,
                        "train/grad_norm": average_grad_norm,
                        "train/lr": optimizer.param_groups[0]["lr"],
                    }
                    metrics = flatten_dict(
                        logger_dict,
                        sep="/",
                    )
                    accelerator.log(metrics, step=train_steps)

                    # reset losses
                    running_loss = 0
                    running_grad_norm = 0
                    gpu_memory_monitor.reset_peak_stats()

                    # gc cleanup
                    gc.collect()

                # visualize teacher-forcing predictions
                if train_steps % config.checkpoint.visualize_every == 0 and accelerator.is_main_process:
                    logger.info("Visualizing teacher-forcing reconstruction.....")
                    pred_recon_grid, gt_recon_grid = evaluate_reconstruction(
                        vq_model,
                        logits[:, :, :config.model.codebook_size],
                        tokens,
                        config.checkpoint.visualize_num,
                        image_size=config.dataset.image_size,
                        codebook_embed_dim=config.model.codebook_embed_dim,
                        device=device
                    )
                    accelerator.log(
                        {
                            "pred_recon": wandb.Image(pred_recon_grid),
                            "gt_recon": wandb.Image(gt_recon_grid),
                        },
                        step=train_steps,
                    )

        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            save_with_retries(accelerator, logger, checkpoint_dir, model, epoch=epoch)

    # final checkpoint
    save_with_retries(accelerator, logger, checkpoint_dir, model, final=True)

    # end training
    logger.info("Training Done.")
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--uuid", type=str)
    parser.add_argument("--config", type=str, default="configs/global_model_entropy.yaml")

    # debug mode
    parser.add_argument("--prod", action="store_true")

    # results dir
    parser.add_argument("--results_dir", type=str)

    # resume dir
    parser.add_argument("--resume-dir", type=str, default=None)

    # model
    parser.add_argument("--model.gpt_model", type=str)
    parser.add_argument("--model.attn_impl", type=str, default="xformers")

    # entropy model
    parser.add_argument("--patcher.entropy_model_checkpoint_config", type=str)
    parser.add_argument("--patcher.entropy_model_checkpoint", type=str)
    parser.add_argument("--patcher.threshold", type=float)
    parser.add_argument("--patcher.max_patch_length", type=int)
    parser.add_argument("--patcher.patch_size", type=float)


    # training arguments
    parser.add_argument("--training.run_patcher", action="store_true")
    parser.add_argument("--training.packed_inputs", action="store_true")
    parser.add_argument("--training.use_ca_rope", action="store_true")
    parser.add_argument("--training.gradient_checkpointing", action="store_true")
    parser.add_argument("--training.num_workers", type=int)
    parser.add_argument("--training.global_batch_size", type=int)
    parser.add_argument("--training.epochs", type=int)
    
    # optimizer arguments
    parser.add_argument("--optimizer.lr", type=float)

    # visualization arguments
    parser.add_argument("--checkpoint.log_every", type=int)
    parser.add_argument("--checkpoint.visualize_every", type=int)
    parser.add_argument("--checkpoint.visualize_num", type=int)
    
    # accelerator arguments
    parser.add_argument("--accelerator.gradient_accumulation_steps", type=int)

    # parse arguments
    config = parser.parse_args()
    args_dict = {k: v for k, v in vars(config).items() if v is not None}
    cli_cfg = OmegaConf.from_dotlist([f"{k}={v}" for k, v in args_dict.items()])

    # load omegaconfig
    config = OmegaConf.load(config.config)

    # merge with script args
    config = OmegaConf.merge(config, cli_cfg)

    # wandb offline
    os.environ["WANDB__SERVICE_WAIT"] = "600"
    if config.training.wandb.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    config = setup_default_config_values(config)

    # ensure that we are not using sdpa attention with torch.compile
    if config.accelerator.inductor_mode is not None:
        if config.model.attn_impl == "sdpa":
            print("WARNING: SDPA does not work correctly with torch.compile for torch versions greater than 2.3.1")
            raise Exception("SDPA is not supported")

    # debug arguments
    if not config.prod:
        print(
            "----------------------------------------------- RUNNING IN DEBUG MODE ----------------------------------------------"
        )
        config.results_dir = f"{config.results_dir}_debug"
        config.training.global_batch_size = 256
        config.checkpoint.visualize_every = 2000

    if config.resume_dir:
        print(f"--------------- Resuming training from {config.resume_dir} -----------")

    main(config, uuid=config.uuid, debug=not config.prod)
