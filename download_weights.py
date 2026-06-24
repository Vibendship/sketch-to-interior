"""Download all model weights during Docker build."""
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

MODELS = [
    "SG161222/RealVisXL_V5.0_Lightning",
    "diffusers/controlnet-depth-sdxl-1.0",
    "diffusers/controlnet-canny-sdxl-1.0",
    "madebyollin/sdxl-vae-fp16-fix",
    "Intel/dpt-large",
    "lllyasviel/Annotators",
]

for model in MODELS:
    print(f"Downloading {model}...")
    try:
        snapshot_download(model)
    except Exception as e:
        print(f"  Warning: {e}")

print("All weights pre-downloaded")
