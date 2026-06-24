#!/usr/bin/env python3
"""Pre-download all model weights during Docker build so setup() is fast.
Must match predict.py's from_pretrained args to cache the right files."""

from transformers import AutoImageProcessor, DPTForDepthEstimation
from controlnet_aux import HEDdetector, PidiNetDetector, LineartDetector, MLSDdetector

print("[1/5] Depth model (DPT-Hybrid) …")
AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")

print("[2/5] ControlNet depth (fp16) …")
from diffusers import ControlNetModel
ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0")

print("[3/5] ControlNet canny (fp16) …")
ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0")

print("[4/5] RealVisXL V5 Lightning (fp16, safetensors) …")
from diffusers import StableDiffusionXLControlNetPipeline
StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    variant="fp16",
    use_safetensors=True,
)

print("[5/5] Annotators …")
HEDdetector.from_pretrained("lllyasviel/Annotators")
PidiNetDetector.from_pretrained("lllyasviel/Annotators")
LineartDetector.from_pretrained("lllyasviel/Annotators")
MLSDdetector.from_pretrained("lllyasviel/Annotators")

print("All models downloaded and cached.")
