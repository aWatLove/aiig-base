import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI(title="Image Generation Service", description="Diffusion-based image generation")


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    user_id: Optional[str] = None
    height: int = 512
    width: int = 512
    steps: int = 20


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5")
_PIPELINE: Optional[StableDiffusionPipeline] = None


def get_pipeline() -> StableDiffusionPipeline:
    """
    Ленивая инициализация пайплайна Stable Diffusion.
    Модель загружается при первом запросе и переиспользуется дальше.
    """
    global _PIPELINE
    if _PIPELINE is None:
        dtype = torch.float16 if DEVICE == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=dtype,
        )
        pipe = pipe.to(DEVICE)
        pipe.safety_checker = None  # для исследовательских целей, этический анализ в дипломе
        _PIPELINE = pipe
    return _PIPELINE


OUTPUT_DIR = Path(os.environ.get("IMAGE_OUTPUT_DIR", "generated"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/generate")
async def generate(req: GenerateRequest):
    """
    Эндпоинт генерации:
    - использует единый пайплайн Stable Diffusion;
    - генерирует одно изображение и сохраняет его в локальную директорию;
    - возвращает относительный путь/URL и использованные параметры.
    """
    pipe = get_pipeline()

    with torch.autocast(DEVICE) if DEVICE == "cuda" else torch.no_grad():
        result = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            num_inference_steps=req.steps,
            height=req.height,
            width=req.width,
        )
    image = result.images[0]

    # Простое именование файлов по счётчику
    existing = list(OUTPUT_DIR.glob("image_*.png"))
    next_idx = len(existing) + 1
    filename = OUTPUT_DIR / f"image_{next_idx:05d}.png"
    image.save(filename)

    image_url = f"/static/{filename.name}"  # в дальнейшем можно повесить StaticFiles

    return {
        "image_url": image_url,
        "used_prompt": req.prompt,
        "height": req.height,
        "width": req.width,
        "steps": req.steps,
        "device": DEVICE,
        "model_id": MODEL_ID,
    }


