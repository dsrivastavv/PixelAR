import io

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import torch.nn.functional as F


def pltToPIL(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def concatenate_images_horizontally(img1, img2):
    """
    Concatenates two PIL images horizontally and returns the result.
    Assumes both images are the same height or resizes them to match.
    """
    # Resize images to the same height if needed
    if img1.height != img2.height:
        new_height = min(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * new_height / img1.height), new_height))
        img2 = img2.resize((int(img2.width * new_height / img2.height), new_height))

    # Create a new image with combined width
    combined_width = img1.width + img2.width
    combined_img = Image.new("RGBA", (combined_width, img1.height))

    # Paste the images side by side
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (img1.width, 0))

    return combined_img


def make_grid(imgs: np.ndarray, scale=0.5, row_first=True):
    """
    Args:
        imgs: [B, H, W, C] in [0, 1]
    Output:
        x row of images, and 2x column of images
        which means 2 x ^ 2 <= B

        img_grid: np.ndarray, [H', W', C]
    """

    B, H, W, C = imgs.shape
    imgs = torch.tensor(imgs)
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    num_row = int(np.sqrt(B / 2))
    if num_row < 1:
        num_row = 1
    num_col = int(np.ceil(B / num_row))

    if row_first:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_col, padding=0)
    else:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_row, padding=0)

    img_grid = img_grid.permute(1, 2, 0).cpu().numpy()

    # resize by scale
    img_grid = cv2.resize(img_grid, None, fx=scale, fy=scale)
    return img_grid

def decode_codes_to_img(vq_model, codes, tgt_size, codebook_embed_dim):
    qz_shape = (
        codes.shape[0],
        codebook_embed_dim,
        int(codes.shape[1] ** 0.5),
        int(codes.shape[1] ** 0.5)
    )
    results = vq_model.decode_code(codes, qz_shape)
    if results.shape[-1] != tgt_size:
        results = F.interpolate(results, size=(tgt_size, tgt_size), mode="bicubic")
    imgs = results.detach() * 127.5 + 128
    imgs = torch.clamp(imgs, 0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return imgs