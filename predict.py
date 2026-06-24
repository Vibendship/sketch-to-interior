import os
import cv2
import numpy as np
from PIL import Image
import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
from transformers import AutoTokenizer, DPTFeatureExtractor, DPTForDepthEstimation
from controlnet_aux import (
    CannyDetector,
    HEDdetector,
    PidiNetDetector,
    LineartDetector,
    MLSDdetector,
    OpenposeDetector,
)

# ── Sketch detectors ──────────────────────────────────────────────────

DETECTORS = {
    "HED": lambda: HEDdetector.from_pretrained("lllyasviel/Annotators"),
    "PidiNet": lambda: PidiNetDetector.from_pretrained("lllyasviel/Annotators"),
    "Lineart": lambda: LineartDetector.from_pretrained("lllyasviel/Annotators"),
    "Canny": lambda: CannyDetector(),
    "MLSD": lambda: MLSDdetector.from_pretrained("lllyasviel/Annotators"),
}


def combine_detectors(img: Image.Image, primary: str, secondary: str, w1: float, w2: float):
    """Run two detectors and blend their outputs, like HedPidNet."""
    p_img = DETECTORS[primary]()(img)
    if secondary:
        s_img = DETECTORS[secondary]()(img)
        p_arr = np.array(p_img, dtype=np.float32)
        s_arr = np.array(s_img, dtype=np.float32)
        blended = (p_arr * w1 + s_arr * w2).clip(0, 255).astype(np.uint8)
        return Image.fromarray(blended)
    return p_img


def preprocess_sketch(
    img: Image.Image,
    sketch_type: str,
    blur_size: int,
    erosion_iters: int,
    dilation_iters: int,
) -> Image.Image:
    """Detect edges / sketch lines and apply morphological cleanup."""

    # ── built-in composite types ──
    composite = {
        "HedPidNet": ("HED", "PidiNet", 0.6, 0.5),
        "CannyPidNet": ("Canny", "PidiNet", 0.7, 0.5),
        "CannyHed": ("Canny", "HED", 0.7, 0.5),
    }

    if sketch_type in composite:
        p, s, w1, w2 = composite[sketch_type]
        sketch = combine_detectors(img, p, s, w1, w2)
    elif sketch_type in DETECTORS:
        sketch = DETECTORS[sketch_type]()(img)
    else:
        sketch = img.copy()

    # ── convert to grayscale if needed ──
    if sketch.mode != "L":
        sketch = sketch.convert("L")

    arr = np.array(sketch)

    # ── blur ──
    if blur_size > 1 and blur_size % 2 == 1:
        arr = cv2.GaussianBlur(arr, (blur_size, blur_size), 0)

    # ── invert if white-on-black (lineart style) ──
    # HED/PidiNet output black lines on white; ensure consistency
    if np.mean(arr) > 127:
        arr = 255 - arr

    # ── morphology ──
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if erosion_iters > 0:
        arr = cv2.erode(arr, kernel, iterations=erosion_iters)
    if dilation_iters > 0:
        arr = cv2.dilate(arr, kernel, iterations=dilation_iters)

    return Image.fromarray(arr).convert("RGB")


def get_depth_map(image: Image.Image, feature_extractor, depth_estimator, device):
    """Generate a depth map using DPT-Large."""
    image_np = np.array(image)
    if image_np.ndim == 2:
        image_np = np.stack([image_np] * 3, axis=-1)
    pil = Image.fromarray(image_np)

    inputs = feature_extractor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad():
        depth_map = depth_estimator(**inputs).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth_map = depth_map.cpu().numpy()
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_map = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    depth_map = (depth_map * 255).astype(np.uint8)
    return Image.fromarray(depth_map).convert("RGB")


# ── Schedulers ────────────────────────────────────────────────────────

SCHEDULERS = {
    "DPM++ 2M Karras": lambda config: DPMSolverMultistepScheduler.from_config(
        config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
    ),
    "Euler a": lambda config: EulerAncestralDiscreteScheduler.from_config(config),
}


# ── Predictor ─────────────────────────────────────────────────────────

MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"
CONTROL_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
CONTROL_CANNY_ID = "diffusers/controlnet-canny-sdxl-1.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"


class Predictor(BasePredictor):
    def setup(self):
        device = "cuda"
        self.device = device

        print("Loading depth estimator …")
        self.depth_feature_extractor = DPTFeatureExtractor.from_pretrained(
            "Intel/dpt-large"
        )
        self.depth_estimator = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-large"
        ).to(device)

        print("Loading ControlNets …")
        controlnet_depth = ControlNetModel.from_pretrained(
            CONTROL_DEPTH_ID, torch_dtype=torch.float16
        ).to(device)
        controlnet_canny = ControlNetModel.from_pretrained(
            CONTROL_CANNY_ID, torch_dtype=torch.float16
        ).to(device)

        print("Loading VAE …")
        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=torch.float16).to(
            device
        )

        print("Loading RealVisXL V5 Lightning pipeline …")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_ID,
            controlnet=[controlnet_depth, controlnet_canny],
            vae=vae,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        ).to(device)

        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()

        print("Loading Compel …")
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

    def predict(
        self,
        image: Path = Input(description="Input sketch or outline"),
        prompt: str = Input(
            default="masterfully designed interior, photorealistic, interior design magazine quality, 8k uhd, highly detailed"
        ),
        negative_prompt: str = Input(
            default="ugly, deformed, noisy, blurry, low quality, glitch, distorted, disfigured, bad proportions, duplicate, out of frame, watermark, signature, text, bad hands, bad anatomy, sketch, line art, cartoon"
        ),
        sketch_type: str = Input(
            default="HedPidNet",
            choices=[
                "PidiNet",
                "HED",
                "Lineart",
                "Canny",
                "CannyPidNet",
                "CannyHed",
                "HedPidNet",
                "MLSD",
                "none",
            ],
        ),
        depth_strength: float = Input(
            default=0.8, ge=0, le=2, description="Depth ControlNet strength"
        ),
        edge_strength: float = Input(
            default=0.85, ge=0, le=2, description="Edge ControlNet strength"
        ),
        guidance_scale: float = Input(default=7.5, ge=1, le=30),
        steps: int = Input(default=6, ge=1, le=50),
        blur_size: int = Input(
            default=3, description="Gaussian blur kernel (odd number)"
        ),
        erosion_iterations: int = Input(default=2, ge=0, le=20),
        dilation_iterations: int = Input(default=5, ge=0, le=20),
        seed: int = Input(default=-1, description="Random seed. -1 = random"),
        width: int = Input(default=1024, ge=512, le=2048),
        height: int = Input(default=1024, ge=512, le=2048),
    ) -> Path:
        if seed == -1:
            seed = np.random.randint(0, 2**31)
        generator = torch.manual_seed(seed)

        # ── Load input ──
        img = Image.open(image).convert("RGB")
        img = img.resize((width, height))

        # ── Sketch detection ──
        if sketch_type == "none":
            sketch_img = img.copy()
        else:
            sketch_img = preprocess_sketch(
                img, sketch_type, blur_size, erosion_iterations, dilation_iterations
            )

        # ── Depth map ──
        depth_img = get_depth_map(
            img, self.depth_feature_extractor, self.depth_estimator, self.device
        )

        # ── Conditioning images ──
        # ControlNet 0 = depth, ControlNet 1 = edge
        controlnet_images = [depth_img, sketch_img]
        controlnet_scales = [depth_strength, edge_strength]
        # controlnet_conditioning_scale in SDXL pipeline expects a list per ControlNet

        # ── Encode prompt ──
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = self.compel(prompt, negative_prompt)

        # ── Run ──
        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=img,
            controlnet_conditioning_image=controlnet_images,
            controlnet_conditioning_scale=controlnet_scales,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            width=width,
            height=height,
            output_type="pil",
        ).images[0]

        out_path = "/tmp/output.png"
        result.save(out_path)
        return Path(out_path)
