"""Minimal OpenAI-compatible (magic666 gateway) client: text, vision and image-editing (Nano Banana) calls.

Environment: MAGIC666_API_KEY (required), MAGIC666_BASE_URL (default https://magic666.top/v1),
MAGIC666_TEXT_MODEL / MAGIC666_VISION_MODEL / MAGIC666_IMAGE_MODEL (comma-separated fallback lists),
MAGIC666_TIMEOUT, MAGIC666_RETRIES. Any gateway serving /chat/completions with image outputs works.
"""

import base64
import json
import mimetypes
import os
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image

BASE_URL = os.getenv("MAGIC666_BASE_URL", "https://magic666.top/v1").rstrip("/")
TEXT_MODEL = os.getenv("MAGIC666_TEXT_MODEL", "gemini-2.5-pro")
VISION_MODEL = os.getenv("MAGIC666_VISION_MODEL", TEXT_MODEL)
IMAGE_MODEL = os.getenv("MAGIC666_IMAGE_MODEL", "gemini-2.5-flash-image")
TIMEOUT = int(os.getenv("MAGIC666_TIMEOUT", "300"))
RETRIES = int(os.getenv("MAGIC666_RETRIES", "3"))


def _with_retries(fn: Callable[[], Any]) -> Any:
    # The gateway intermittently returns HTTP 429/5xx or empty `content_filter` completions.
    for attempt in range(1, max(1, RETRIES) + 1):
        try:
            return fn()
        except (RuntimeError, requests.RequestException) as error:
            if attempt == RETRIES:
                raise
            print(f"[magic666] attempt {attempt}/{RETRIES} failed: {str(error)[:300]}", flush=True)
            time.sleep(min(2 * attempt, 8))


def _api_key() -> str:
    key = os.getenv("MAGIC666_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Set MAGIC666_API_KEY (OpenAI-compatible gateway key, see README).")
    return key


def _image_to_data_url(image: str | Path | Image.Image) -> str:
    if isinstance(image, Image.Image):
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

    path = Path(image).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    if mime is None:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('utf-8')}"


def _request(model: str, prompt: str, images: list[str | Path | Image.Image] | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images or []:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(image)}})
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        json={"model": model, "messages": [{"role": "user", "content": content}], "stream": False},
        timeout=TIMEOUT,
    )
    if response.status_code != 200:
        raise RuntimeError(f"magic666 request failed: HTTP {response.status_code}\n{response.text[:2000]}")
    return response.json()


def _extract_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for choice in data.get("choices", []) or []:
        message = choice.get("message", {}) or {}
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    text = "\n".join(chunk for chunk in chunks if chunk).strip()
    if not text:
        raise RuntimeError("No text found in magic666 response: " + json.dumps(data, ensure_ascii=False)[:2000])
    return text


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for item in data.get("data", []) or []:
        if isinstance(item, dict):
            if item.get("url"):
                urls.append(str(item["url"]))
            elif item.get("b64_json"):
                urls.append("data:image/png;base64," + str(item["b64_json"]))
    if urls:
        return urls
    for choice in data.get("choices", []) or []:
        message = choice.get("message", {}) or {}
        content = message.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        urls.append(str(url))
        elif isinstance(content, str):
            urls.extend(re.findall(r"https?://[^\s)\"'<>]+\.(?:png|jpg|jpeg|webp)", content, re.I))
        for image in message.get("images", []) or []:
            if isinstance(image, str):
                urls.append(image)
            elif isinstance(image, dict) and image.get("url"):
                urls.append(str(image["url"]))
    return urls


def _download_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _with_fallback(models: str, call: Callable[[str], Any]) -> Any:
    """`models` is a comma-separated list (MAGIC666_*_MODEL); each model gets RETRIES attempts, then the next
    one is tried. Lets the pipeline survive a model whose gateway channel is down or unavailable."""
    names = [m.strip() for m in models.split(",") if m.strip()]
    for i, name in enumerate(names):
        try:
            return _with_retries(lambda: call(name))
        except (RuntimeError, requests.RequestException) as error:
            if i == len(names) - 1:
                raise
            print(f"[magic666] model {name} failed ({str(error)[:200]}); falling back to {names[i + 1]}", flush=True)


def generate_text(prompt: str, model: str = TEXT_MODEL) -> str:
    return _with_fallback(model, lambda m: _extract_text(_request(m, prompt)))


def generate_multimodal_text(prompt: str, images: list[str | Path | Image.Image], model: str = VISION_MODEL) -> str:
    return _with_fallback(model, lambda m: _extract_text(_request(m, prompt, images)))


def _generate_image_once(prompt: str, images: list[str | Path | Image.Image] | None, model: str) -> Image.Image:
    data = _request(model, prompt, images)
    urls = _extract_image_urls(data)
    if not urls:
        raise RuntimeError("No image URL found in magic666 response: " + json.dumps(data, ensure_ascii=False)[:2000])
    return _download_image(urls[0])


def generate_image(prompt: str, images: list[str | Path | Image.Image] | None = None, model: str = IMAGE_MODEL) -> Image.Image:
    return _with_fallback(model, lambda m: _generate_image_once(prompt, images, m))
