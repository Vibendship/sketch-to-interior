# Sketch to Interior — Cog Model

Takes a sketch/outline and generates a photorealistic interior render using SDXL + RealVisXL V5.0 + ControlNet.

## How it works

1. **Input**: a sketch, line art, edge map, or photo
2. **Sketch detection**: extracts edges using the selected detector (HED, Canny, Lineart, PidiNet, etc.)
3. **Cleanup**: applies blur, erosion, dilation to clean the sketch
4. **Dual ControlNet**: uses both Depth + Edge ControlNet to preserve geometry
5. **SDXL + RealVisXL V5**: generates photorealistic interior

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `image` | required | Input sketch or outline |
| `prompt` | `"masterfully designed interior..."` | Style description |
| `negative_prompt` | defaults | What to avoid |
| `sketch_type` | `HedPidNet` | Edge detector: PidiNet, HED, Lineart, Canny, CannyPidNet, CannyHed, HedPidNet, MLSD |
| `depth_strength` | 0.8 | How strongly depth guides the output |
| `edge_strength` | 0.85 | How strongly edge sketch guides the output |
| `guidance_scale` | 7.5 | CFG scale |
| `steps` | 6 | Inference steps |
| `blur_size` | 3 | Clean up noisy sketches (odd number) |
| `erosion_iterations` | 2 | Thin sketch lines |
| `dilation_iterations` | 5 | Thicken sketch lines |
