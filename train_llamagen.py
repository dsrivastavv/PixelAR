# Modified from:
#   LLaMAGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/train/extract_codes_t2i.py
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import gc
import torch

from dynamic_tokenization.dataset.imagenet import CustomDataset
from dynamic_tokenization.dataset.patch_dataset import PatchDataset
from dynamic_tokenization.evaluation.evaluate import evaluate_reconstruction
from dynamic_tokenization.models.gpt import GPT_models
from dynamic_tokenization.models.utils import save_with_retries
from dynamic_tokenization.tokenizer_image.vq_model import VQ_models
from dynamic_tokenization.utils.gpu_monitor import GPUMemoryMonitor
from dynamic_tokenization.utils.logger import create_logger

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


def build_imagenet_code(code_path, image_size):
    feature_dir = f"{code_path}/imagenet{image_size}_codes"
    label_dir = f"{code_path}/imagenet{image_size}_labels"
    assert os.path.exists(feature_dir) and os.path.exists(
        label_dir
    ), f"please first run: bash scripts/autoregressive/extract_codes_c2i.sh ..."
    return CustomDataset(feature_dir, label_dir)


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
        project_name="LLamaGen",
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
    # setup dataset
    dataset = build_imagenet_code(config.dataset.code_path, config.dataset.image_size)
    flip_info = "with" if dataset.flip else "without"
    aug_info = 10 if "ten_crop" in dataset.feature_dir else 1
    aug_info = 2 * aug_info if dataset.aug_feature_dir is not None else aug_info
    logger.info(
        f"Dataset contains {len(dataset):,} images ({config.dataset.code_path}) "
        f"{flip_info} flip augmentation and {aug_info} crop augmentation"
    )
    if config.training.packed_inputs:
        logger.info("Running with packed inputs...")
        dataset = PatchDataset(dataset)
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
        persistent_workers=True,
        prefetch_factor=8,
    )

    # setup model
    if config.model.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = config.model.dropout_p
    latent_size = config.dataset.image_size // config.model.downsample_size
    assert config.model.cls_token_num >0, "class token number must be greater than 0 for class-conditional generation"
    model = GPT_models[config.model.gpt_model](
        vocab_size=config.model.codebook_size,
        block_size=latent_size**2,
        num_classes=config.dataset.num_classes,
        cls_token_num=config.model.cls_token_num,
        model_type=config.model.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=config.model.drop_path_rate,
        token_dropout_p=config.model.token_dropout_p,
        packed_inputs=config.training.packed_inputs
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")

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

    # prepare with accelerate
    model.move_buffers_to_device(device)
    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    ################## Resume Training ##################
    if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
        saved_ckpt_dirs = [_ for _ in os.listdir(checkpoint_dir) if _.startswith("epoch")]
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        ckpt_dir = f"{checkpoint_dir}/{saved_ckpt_dirs[-1]}/"
        logger.info(f"Resuming from {ckpt_dir}")
        accelerator.load_state(ckpt_dir)
        start_epoch = int(saved_ckpt_dirs[-1].split("_")[-1])+1
        train_steps = int((start_epoch) * len(data_loader))
    else:
        start_epoch = 0
        train_steps = 0

    #################### Training Loop ####################
    num_epochs = config.training.epochs
    running_loss, running_grad_norm, start_time = 0, 0, time.time()
    logger.info(f"Training for {num_epochs} epochs starting at epoch {start_epoch}...")
    accelerator.wait_for_everyone()
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for batch in data_loader:
            model.train()
            tokens, labels = batch[0], batch[1]
            
            with accelerator.accumulate(model):
                idx = tokens[:, :-1]
                if config.training.packed_inputs:
                    seqlens = torch.tensor([idx.shape[1]]*idx.shape[0])
                    logits, loss = model(
                        idx=idx.contiguous().view(-1), 
                        cond_idx=labels,
                        targets=tokens.contiguous().view(-1),
                        seqlens=seqlens
                    )
                else:
                    logits, loss = model(
                        idx=idx, 
                        cond_idx=labels,
                        targets=tokens,
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
                logits = logits.detach()
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

                    # gpu monitor
                    gpu_mem_stats = gpu_memory_monitor.get_peak_stats(logger=logger)

                    # speed
                    end_time = time.time()
                    average_time = (end_time - start_time) / config.checkpoint.log_every
                    start_time = time.time()

                    # log info
                    logger.info(
                        f"Epoch {epoch:08d} | Step {train_steps:08d} | Loss {average_loss:.4f} | Time {average_time:.4f}s | Grad Norm {average_grad_norm:.4f} | LR {optimizer.param_groups[0]['lr']:.5f} |  mem: {gpu_mem_stats.max_reserved_gib:.0f}% | pow: {gpu_mem_stats.power_draw/1000} W"
                    )
                    logger_dict = {
                        "train/loss": average_loss,
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
                        logits,
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

        # checkpointing
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
    parser.add_argument("--config", type=str, default="configs/baselines/llamagen.yaml")

    # debug mode
    parser.add_argument("--prod", action="store_true")

    # results dir
    parser.add_argument("--results_dir", type=str)

    # resume dir
    parser.add_argument("--resume-dir", type=str, default=None)

    # model args
    parser.add_argument("--model.gpt_model", type=str)
    parser.add_argument("--model.attn_impl", type=str, default="xformers")

    # training arguments
    parser.add_argument("--training.packed_inputs", action="store_true")
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
        
    # setup default config values
    if 'uuid' not in config:
        print("setting uuid to None")
        config['uuid'] = None

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
        config.checkpoint.visualize_every = 500

    if config.resume_dir:
        print(f"--------------- Resuming training from {config.resume_dir} -----------")

    main(config, uuid=config.uuid, debug=not config.prod)
