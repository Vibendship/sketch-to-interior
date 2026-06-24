import gc
import os
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DDIMScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
from transformers import AutoImageProcessor, DPTForDepthEstimation
from controlnet_aux import (
    CannyDetector,
    HEDdetector,
    PidiNetDetector,
    LineartDetector,
    MLSDdetector,
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

    if sketch.mode != "L":
        sketch = sketch.convert("L")
    arr = np.array(sketch)

    if blur_size > 1 and blur_size % 2 == 1:
        arr = cv2.GaussianBlur(arr, (blur_size, blur_size), 0)
    if np.mean(arr) > 127:
        arr = 255 - arr

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    if erosion_iters > 0:
        arr = cv2.erode(arr, kernel, iterations=erosion_iters)
    if dilation_iters > 0:
        arr = cv2.dilate(arr, kernel, iterations=dilation_iters)

    return Image.fromarray(arr).convert("RGB")


def get_depth_map(image: Image.Image, processor, model, device):
    """Depth map using lightweight DPT-Hybrid."""
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        depth = model(**inputs).predicted_depth

    depth = F.interpolate(
        depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    depth = depth.cpu().numpy()
    dmin, dmax = depth.min(), depth.max()
    depth = (depth - dmin) / (dmax - dmin + 1e-8)
    depth = (depth * 255).astype(np.uint8)
    return Image.fromarray(depth).convert("RGB")


# ── Models ────────────────────────────────────────────────────────────

MODEL_ID = "SG161222/RealVisXL_V5.0_Lightning"
CONTROL_DEPTH_ID = "diffusers/controlnet-depth-sdxl-1.0"
CONTROL_CANNY_ID = "diffusers/controlnet-canny-sdxl-1.0"


class Predictor(BasePredictor):
    def setup(self):
        device = "cuda"
        self.device = device

        # ── 1. Lightweight depth model ──
        print("Loading depth model (DPT-Hybrid) …")
        self.depth_processor = AutoImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        self.depth_model = DPTForDepthEstimation.from_pretrained(
            "Intel/dpt-hybrid-midas", low_cpu_mem_usage=True
        ).to(device)
        torch.cuda.empty_cache()
        gc.collect()

        # ── 2. ControlNets ──
        print("Loading ControlNet depth …")
        controlnet_depth = ControlNetModel.from_pretrained(
            CONTROL_DEPTH_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        torch.cuda.empty_cache()
        gc.collect()

        print("Loading ControlNet canny …")
        controlnet_canny = ControlNetModel.from_pretrained(
            CONTROL_CANNY_ID,
            torch_dtype=torch.float16,
            variant="fp16",
            low_cpu_mem_usage=True,
        ).to(device)
        torch.cuda.empty_cache()
        gc.collect()

        # ── 3. Main pipeline ──
        print("Loading RealVisXL V5 Lightning pipeline …")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_ID,
            controlnet=[controlnet_depth, controlnet_canny],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        self.pipe = self.pipe.to(device)
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        gc.collect()

        # ── 4. Compel ──
        print("Loading Compel …")
        self.compel = Compel(
            tokenizer=[self.pipe.tokenizer, self.pipe.tokenizer_2],
            text_encoder=[self.pipe.text_encoder, self.pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        print("Setup complete.")

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
                "PidiNet", "HED", "Lineart", "Canny",
                "CannyPidNet", "CannyHed", "HedPidNet", "MLSD", "none",
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
        blur_size: int = Input(default=3, description="Gaussian blur kernel (odd number)"),
        erosion_iterations: int = Input(default=2, ge=0, le=20),
        dilation_iterations: int = Input(default=5, ge=0, le=20),
        seed: int = Input(default=-1, description="Random seed. -1 = random"),
        width: int = Input(default=1024, ge=512, le=2048),
        height: int = Input(default=1024, ge=512, le=2048),
    ) -> Path:
        if seed == -1:
            seed = np.random.randint(0, 2**31)
        generator = torch.manual_seed(seed)

        # ── Load & resize input ──
        img = Image.open(image).convert("RGB")
        img = img.resize((width, height))

        # ── Sketch / edge detection ──
        if sketch_type == "none":
            sketch_img = img.copy()
        else:
            sketch_img = preprocess_sketch(
                img, sketch_type, blur_size, erosion_iterations, dilation_iterations
            )

        # ── Depth map ──
        depth_img = get_depth_map(img, self.depth_processor, self.depth_model, self.device)

        # ── Encode prompt ──
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.compel(prompt, negative_prompt)

        # ── Run ──
        result = self.pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=img,
            controlnet_conditioning_image=[depth_img, sketch_img],
            controlnet_conditioning_scale=[depth_strength, edge_strength],
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
