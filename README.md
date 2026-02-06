# Dynamic Autoregressive Image Generation

## Abstract

Autoregressive Image generation with transformer-based models is constrained by tokenization strategies that lead to a quadratic increase in token count with increasing image resolution. We propose **DyART** (Dynamic Autoregressive Transformer), a novel approach for reducing token count by leveraging spatial redundancy in images to dynamically merge tokens into larger units, or patches, guided by next-token prediction entropy. Our method achieves a reduced sub-quadratic growth in patch count with increasing resolution on the ImageNet benchmark, demonstrating improved computational efficiency. Further, we introduce a ROPE-based positional encoding scheme for dynamically sized patches, capturing spatial proximity across variable-length patches. Finally, we validate the scalability of our approach by training models from 120M to 1.3B parameters, achieving faster convergence and improved FID on the ImageNet benchmark. Our approach can be integrated with existing discrete 2D tokenizers, making it compatible with existing autoregressive image generation architectures and opening new avenues for efficient training of multimodal large language models.

## Quick start

1. Run these commands to setup environment:

```bash
conda create -n llamagen python=3.13
conda activate llamagen
pip install -r requirements.txt
```

2. Symlink to folder containing data, entropy model for patching, and evaluation artifacts:

```bash
ln -s /mnt/sandbox/dsriv/dynamic_tokenization/data data
```

3. Sample few images from DGPT-L(~375M params) model:

```bash
python sample_c2i.py --experiment_dir data/model_checkpoints/global_model_entropy_row_break_b/20250820_13/ --output-dir samples
```

## Training

Run the following command to train DGPT-L model for image resolution 256 on 4x40G GPUs:

```bash
accelerate launch --multi-gpu train_global_model.py --config configs/patching_modes/global_model_entropy_row_break.yaml --model.gpt_model DGPT-L --patcher.entropy_model_checkpoint_config data/entropy_model/20250808_11/config.yaml --patcher.entropy_model_checkpoint data/entropy_model/20250808_11/checkpoints/epoch_00000299/model.safetensors --prod
```

**Note**: We currently have two entropy models under `data/entropy_model` trained for image resolution 256 and 384:

| Resolution | Folder      |
| ---------- | ----------- |
| 256        | 20250808_11 |
| 384        | 20250810_21 |

## Evaluation

1. Generate 50,000 image samples for FID-50K evaluation and pack generated images into a `.npz` file for use with evaluation script:

```bash
accelerate launch sample_c2i_ddp.py --experiment_dir data/model_checkpoints/global_model_entropy_row_break_b/20250820_13/ --cfg-scale 2.0 --output-dir samples
```

2. Run evaluation script to calculate `FID`, `IS`, `sFID`, `precision`, and, `recall`:

```bash
python -m evaluations.c2i.evaluator data/VIRTUAL_imagenet256_labeled.npz samples/epoch-299-topk-0-topp-1.0-temperature-1.0-cfg-2.0-seed-0.npz
```
