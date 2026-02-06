import os
from safetensors.torch import save_file

def save_with_retries(accelerator, logger, checkpoint_dir, model, epoch=None, final=False,
                      max_retries=3, base_delay=1.0):
    name = "final" if final else (f"epoch_{epoch:08d}" if epoch is not None else "final")
    ckpt_path = os.path.join(checkpoint_dir, name)

    # --- primary: Accelerator save_state (if available) ---
    ok = False
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # try saving multiple times with accelerator
        for attempt in range(1, max_retries + 1):
            try:
                accelerator.save_state(ckpt_path)
                logger.info(f"Saved {'final' if final else f'epoch {epoch}'} checkpoint to {ckpt_path} (attempt {attempt})")
                ok = True
                break
            except Exception as e:
                logger.error(f"Error saving {'final' if final else f'epoch {epoch}'} via accelerator: {e} (attempt {attempt}/{max_retries})")

    # --- fallback: only if accelerator saving failed or accelerator is unavailable ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process and not ok:
        os.makedirs(ckpt_path, exist_ok=True)
        
        # save model parameters directly with safetensors
        unwrapped_model = accelerator.unwrap_model(model)
        model_state_dict = unwrapped_model.state_dict()
        
        # save directly
        save_file(model_state_dict, f"{ckpt_path}/model.safetensors")
        logger.info(f"Saved {'final' if final else f'epoch {epoch}'} checkpoint to {ckpt_path} via safetensors fallback")

    return ckpt_path