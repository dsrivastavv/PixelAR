import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from pytorch_fid.fid_score import calculate_fid_given_paths, save_fid_stats
from torchvision import transforms

from dynamic_tokenization.utils.image import (
    center_crop_arr,
    concatenate_images_horizontally,
    decode_codes_to_img,
    make_grid,
    pltToPIL,
)


@torch.no_grad()
def evaluate_reconstruction(
    vq_model, pred_logits, gt_tokens, num_img_visualize, image_size, codebook_embed_dim, no_cond=False, device="cpu"
):
    """Evaluate model reconstruction performance."""
    # get logits of the current batch
    pred_logits = pred_logits[:num_img_visualize]
    gt_tokens = gt_tokens[:num_img_visualize]

    # teacher forcing reconstruction
    start_idx = 0
    img_token_num = pred_logits.shape[1]
    if no_cond:
        pred_recon_indices = torch.zeros(num_img_visualize, img_token_num+1, device=device).long()
        pred_recon_indices[:, 0] = gt_tokens[:, 0]
        for i in range(start_idx, img_token_num):
            pred_recon_indices[:, i+1 : i+2] = torch.argmax(pred_logits[:, i : i + 1], dim=-1)
    else:
        pred_recon_indices = torch.zeros(num_img_visualize, img_token_num, device=device).long()
        for i in range(start_idx, img_token_num):
            pred_recon_indices[:, i : i + 1] = torch.argmax(pred_logits[:, i : i + 1], dim=-1)
    
    pred_recon_imgs = decode_codes_to_img(
        vq_model,
        pred_recon_indices,
        image_size,
        codebook_embed_dim=codebook_embed_dim,
    )
    pred_recon_grid = make_grid(pred_recon_imgs)

    # vq reconstruction
    gt_recon_imgs = decode_codes_to_img(
        vq_model,
        gt_tokens,
        image_size,
        codebook_embed_dim=codebook_embed_dim,
    )
    gt_recon_grid = make_grid(gt_recon_imgs)

    return pred_recon_grid, gt_recon_grid


@torch.no_grad()
def save_images_for_fid(
    vq_model,
    logits,
    config,
    checkpoint_dir,
    train_steps,
    device,
    num_img_visualize,
):
    """Save reconstructed images for FID calculation."""
    ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
    bsz = logits.shape[0]
    assert bsz % num_img_visualize == 0, "visualize_num must be a multiple of per gpu batch size"
    save_dir = os.path.join(ckpt_path, "pred_recon_imgs")
    os.makedirs(save_dir, exist_ok=True)

    start = 0
    img_token_num = logits.shape[1]
    for i in range(bsz // num_img_visualize):
        visualize_logits = logits[start : start + num_img_visualize]
        pred_recon_indices = torch.zeros(num_img_visualize, img_token_num, device=device).long()
        for j in range(img_token_num):
            pred_recon_indices[:, j : j + 1] = torch.argmax(visualize_logits[:, j : j + 1], dim=-1)
        pred_recon_imgs = decode_codes_to_img(
            vq_model,
            pred_recon_indices,
            config.dataset.image_size,
            codebook_embed_dim=config.model.codebook_embed_dim,
        )
        start = start + num_img_visualize

        # Save images
        for img_idx, img_array in enumerate(pred_recon_imgs):
            img = Image.fromarray(img_array)
            save_path = os.path.join(save_dir, f"{img_idx}.png")
            img.save(save_path)

    return save_dir, ckpt_path


@torch.no_grad()
def calculate_and_log_fid(save_dir, ckpt_path, device, accelerator, train_steps, logger):
    """Calculate FID score and log it."""
    npz_dir = os.path.join(ckpt_path, "fid.npz")
    baseline_npz_dir = "data/imagenet_val_fid_stats.npz"
    save_fid_stats([save_dir, npz_dir], batch_size=32, device=device, dims=2048)
    fid_value = calculate_fid_given_paths(
        [baseline_npz_dir, npz_dir],
        batch_size=50,
        dims=2048,
        device=device,
    )
    logger.info(f"FID score: {fid_value}")
    fid_logger_dict = {"train/fid": float(fid_value)}
    accelerator.log(fid_logger_dict, step=train_steps)

    return fid_value


@torch.no_grad()
def analyze_entropy_for_image(
    ref_images_logits,
    ref_image_idx,
    ref_img_name,
    entropy_save_dir,
    ref_images,
    logger,
    entropy_threshold,
):
    """Analyze entropy for a single reference image."""
    # For each reference image:
    # 1. Generate 1-D entropy map
    # 2. Generate 2-D entropy heatmap
    # 3. Generate image and heatmap side by side
    # 4. Print statistics of token entropies
    # get token entropies
    entropy_map_1d = []
    for seq_idx in range(ref_images_logits.shape[1]):
        token_entropy = torch.distributions.categorical.Categorical(
            logits=ref_images_logits[ref_image_idx][seq_idx]
        ).entropy()
        entropy_map_1d.append(token_entropy.item())
    entropy_map_1d = np.array(entropy_map_1d)
    entropy_map_1d = np.concat([[entropy_map_1d.max()], entropy_map_1d])

    # Plotting the entropy values
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(ref_images_logits.shape[1] + 1),
        entropy_map_1d,
        marker="o",
        linestyle="-",
    )
    plt.title(f"Entropy vs Token Number: {ref_img_name}")
    plt.xlabel("Token Number")
    plt.ylabel("Entropy")
    plt.xticks(ticks=range(0, ref_images_logits.shape[1], 16))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{entropy_save_dir}/{ref_img_name}_entropy.png")
    plt.close()

    # heatmap
    entropy_grid = np.array(entropy_map_1d).reshape(16, 16)
    fig = plt.figure(figsize=(2.56, 2.56), dpi=100)
    sns.heatmap(
        entropy_grid,
        cmap="plasma",
        cbar=False,
        xticklabels=False,
        yticklabels=False,
    )
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(
        f"{entropy_save_dir}/{ref_img_name}_heatmap.png",
        bbox_inches="tight",
        pad_inches=0,
    )
    heatmap_img = pltToPIL(fig)
    plt.close()

    # image and heatmap side by side
    combined_image = concatenate_images_horizontally(ref_images[ref_image_idx], heatmap_img)
    combined_image.save(f"{entropy_save_dir}/{ref_img_name}_combined.png")

    # calculate and log statistics
    min_entropy = np.min(entropy_map_1d)
    max_entropy = np.max(entropy_map_1d)
    mean_entropy = np.mean(entropy_map_1d)
    std_entropy = np.std(entropy_map_1d)
    num_above_threshold = np.sum(entropy_map_1d > entropy_threshold)

    logger.info(f"Stats for Ref Image: {ref_img_name}")
    logger.info(f"Minimum entropy: {min_entropy}")
    logger.info(f"Maximum entropy: {max_entropy}")
    logger.info(f"Mean entropy: {mean_entropy}")
    logger.info(f"Standard deviation: {std_entropy}")
    logger.info(f"Number of elements > {entropy_threshold}: {num_above_threshold}")

    # log statistics
    entropy_logger_dict = {
        f"val/{ref_img_name}/min_entropy": float(min_entropy),
        f"val/{ref_img_name}/max_entropy": float(max_entropy),
        f"val/{ref_img_name}/mean_entropy": float(mean_entropy),
        f"val/{ref_img_name}/std_entropy": float(std_entropy),
        f"val/{ref_img_name}/num_above_threshold": int(num_above_threshold),
    }
    return entropy_logger_dict


@torch.no_grad()
def perform_entropy_analysis(model, vq_model, ckpt_path, logger, entropy_threshold=7.0, device="cpu"):
    """Perform entropy analysis on reference images."""
    tokenizer_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    ref_image_paths = [
        "assets/low-detail.jpg",
        "assets/medium-detail.jpg",
        "assets/high-detail.jpg",
    ]
    ref_images = [center_crop_arr(Image.open(path), image_size=256) for path in ref_image_paths]
    ref_images_transformed = torch.stack([tokenizer_transform(image) for image in ref_images], dim=0).to(device)  # type: ignore

    # make save directory
    entropy_save_dir = os.path.join(ckpt_path, "entropy_analysis")
    os.makedirs(entropy_save_dir, exist_ok=True)
    logger.info(f"Entropy visualizations and saving images to {entropy_save_dir}....")

    _, _, info = vq_model.encode(ref_images_transformed)
    ref_images_tokenized = info[2].reshape(len(ref_images), -1)
    cond_idx_dummy = torch.zeros((ref_images_tokenized.shape[0]), device=device)
    ref_images_logits, _ = model.module(
        idx=ref_images_tokenized[:, :-1],
        cond_idx=cond_idx_dummy,
    )

    result_dicts = []
    for ref_image_idx in range(len(ref_image_paths)):
        ref_img_name = ref_image_paths[ref_image_idx].split("/")[1].split(".")[0]
        result_dict = analyze_entropy_for_image(
            ref_images_logits=ref_images_logits,
            ref_image_idx=ref_image_idx,
            ref_img_name=ref_img_name,
            entropy_save_dir=entropy_save_dir,
            ref_images=ref_images,
            logger=logger,
            entropy_threshold=entropy_threshold,
        )
        result_dicts.append(result_dict)

    return result_dicts
