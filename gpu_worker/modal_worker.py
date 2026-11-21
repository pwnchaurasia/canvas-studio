"""Canvas Studio GPU worker: Z-Image-Turbo on a scale-to-zero Modal L4.

Deploy:    modal deploy gpu_worker/modal_worker.py
Prefetch:  modal run gpu_worker/modal_worker.py::prefetch   (CPU-only, fills the weight cache once)

Cost rules baked in: L4 only, no region pinning, scaledown_window=25, no warm containers,
max_containers=1, and edge proxy auth so unauthenticated traffic never wakes a GPU.
"""

import base64
import hmac
import io
import os
import random
import time

import modal

APP_NAME = "canvas-studio"
MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
CACHE_DIR = "/cache"
SECRET_NAME = "canvas-studio-secrets"

# name -> (width, height); all multiples of 16
PRESETS: dict[str, tuple[int, int]] = {
    "9:16": (768, 1344),
    "1:1": (1024, 1024),
    "16:9": (1344, 768),
    "draft": (512, 896),
}
MAX_COUNT = 4
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 4, 12, 9

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "safetensors",
        "peft",
        "hf_transfer",
        "pillow",
        "fastapi[standard]",
        "git+https://github.com/huggingface/diffusers",
    )
    .env({"HF_HOME": CACHE_DIR, "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

weights = modal.Volume.from_name("canvas-studio-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name(SECRET_NAME)]

app = modal.App(APP_NAME)


def _adapter_id() -> str:
    return (os.environ.get("STYLE_ADAPTER_ID") or "").strip()


@app.function(image=image, volumes={CACHE_DIR: weights}, secrets=secrets, timeout=3600)
def prefetch() -> None:
    """Download weights into the Volume on a cheap CPU container (never on a GPU, never in CI)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID)
    if _adapter_id():
        snapshot_download(_adapter_id())
    weights.commit()
    print("Weights cached in volume.")


@app.cls(
    image=image,
    gpu="L4",
    volumes={CACHE_DIR: weights},
    secrets=secrets,
    scaledown_window=25,  # idle GPU billing stops 25s after the last request
    min_containers=0,
    buffer_containers=0,
    max_containers=1,  # hard ceiling: requests queue instead of spawning a 2nd GPU
    timeout=600,
)
class Worker:
    @modal.enter()
    def load(self) -> None:
        import torch
        from diffusers import ZImagePipeline

        t0 = time.perf_counter()
        self.pipe = ZImagePipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda")
        if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_slicing"):
            self.pipe.vae.enable_slicing()

        self.adapter_loaded = False
        adapter = _adapter_id()
        if adapter:
            try:
                self.pipe.load_lora_weights(adapter, adapter_name="custom")
                self.pipe.set_adapters(["custom"], adapter_weights=[0.0])
                self.adapter_loaded = True
                print("Style adapter loaded.")
            except Exception as e:  # bad repo id / incompatible adapter -> run without it
                print(f"Style adapter skipped: {type(e).__name__}: {e}")

        try:
            weights.commit()  # persists anything downloaded on a first-ever run
        except Exception:
            pass
        self.load_s = round(time.perf_counter() - t0, 2)
        self.cold = True
        print(f"Pipeline ready in {self.load_s}s")

    def _render(self, prompt: str, aspect: str, steps: int, count: int, seed: int | None, adapter_scale: float) -> dict:
        import torch

        width, height = PRESETS[aspect]
        if self.adapter_loaded:
            self.pipe.set_adapters(["custom"], adapter_weights=[adapter_scale])

        base_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
        seeds = [(base_seed + i) % 2**32 for i in range(count)]

        t0 = time.perf_counter()
        images = []
        for s in seeds:  # sequential keeps peak VRAM flat on the 24 GB L4
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,  # required for Z-Image-Turbo
                generator=torch.Generator("cuda").manual_seed(s),
            ).images[0]
            buf = io.BytesIO()
            result.save(buf, format="WEBP", quality=90)
            images.append({"seed": s, "image_b64": base64.b64encode(buf.getvalue()).decode()})
        gen_s = round(time.perf_counter() - t0, 2)

        cold, self.cold = self.cold, False
        print(f"Generated {count} x {width}x{height} @ {steps} steps in {gen_s}s (cold={cold})")
        return {
            "images": images,
            "width": width,
            "height": height,
            "generation_seconds": gen_s,
            "cold_start": cold,
            "load_seconds": self.load_s if cold else 0.0,
            "adapter_active": self.adapter_loaded and adapter_scale > 0,
        }

    @modal.asgi_app(requires_proxy_auth=True)  # Modal rejects calls without proxy token at the edge
    def web(self):
        from fastapi import FastAPI, Header, HTTPException
        from pydantic import BaseModel, Field

        class RenderRequest(BaseModel):
            prompt: str = Field(min_length=1, max_length=4000)
            aspect: str = "9:16"
            steps: int = Field(default=STEPS_DEFAULT, ge=STEPS_MIN, le=STEPS_MAX)
            count: int = Field(default=1, ge=1, le=MAX_COUNT)
            seed: int | None = Field(default=None, ge=0, le=2**32 - 1)
            adapter_scale: float = Field(default=0.0, ge=0.0, le=1.0)

        api = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

        @api.post("/generate")
        def generate(req: RenderRequest, x_api_key: str | None = Header(default=None)):
            expected = os.environ.get("STUDIO_API_KEY", "")
            if not expected or not x_api_key or not hmac.compare_digest(x_api_key, expected):
                raise HTTPException(status_code=401, detail="unauthorized")
            if req.aspect not in PRESETS:
                raise HTTPException(status_code=422, detail=f"aspect must be one of {sorted(PRESETS)}")
            return self._render(req.prompt, req.aspect, req.steps, req.count, req.seed, req.adapter_scale)

        return api
