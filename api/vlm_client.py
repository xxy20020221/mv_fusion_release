"""magic666 (OpenAI-compatible) client for VLM / LLM / image-editing calls.

Same calling style as api/magic666_client.py (chat/completions with data-URL images), plus
JSON-only answers validated against a schema and retries on gateway errors. Every call is appended to a JSONL log when
`set_log_path` is used, so VLM decisions can be audited afterwards.
"""

from __future__ import annotations

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
TIMEOUT = int(os.getenv("MAGIC666_TIMEOUT", "600"))
RETRIES = int(os.getenv("MAGIC666_RETRIES", "3"))

_LOG_PATH: Path | None = None

ImageLike = str | Path | Image.Image


class QuotaExhaustedError(Exception):
    """The gateway account has no balance left. Account-wide, so neither retries nor model fallbacks help.
    Deliberately not a RuntimeError, so callers that tolerate a failed call do not swallow it."""


def set_log_path(path: Path | None) -> None:
    global _LOG_PATH
    _LOG_PATH = path
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)


def _log(record: dict[str, Any]) -> None:
    if _LOG_PATH is None:
        return
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _api_key() -> str:
    key = os.getenv("MAGIC666_API_KEY") or os.getenv("OPENAI_API_KEY")
    if key:
        return key
    raise RuntimeError("Set MAGIC666_API_KEY (OpenAI-compatible gateway key, see README).")


def _to_rgb_on_white(image: Image.Image) -> Image.Image:
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def image_to_data_url(image: ImageLike, max_side: int | None = None) -> str:
    """Encode an image as a PNG data URL; transparent backgrounds become white."""
    if not isinstance(image, Image.Image):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        if max_side is None and mimetypes.guess_type(str(path))[0] == "image/jpeg":
            return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("utf-8")
        image = Image.open(path)
    image = _to_rgb_on_white(image)
    if max_side is not None and max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _request(model: str, prompt: str, images: list[ImageLike] | None, max_side: int | None,
             response_json: bool = False, temperature: float | None = None) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images or []:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image, max_side)}})
    payload: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}], "stream": False}
    if response_json:
        payload["response_format"] = {"type": "json_object"}
    if temperature is not None:
        payload["temperature"] = temperature
    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"},
        json=payload,
        timeout=TIMEOUT,
    )
    if response.status_code == 403 and "insufficient_user_quota" in response.text:
        raise QuotaExhaustedError(f"magic666 account quota exhausted - top up the balance and rerun ({response.text[:300]})")
    if response.status_code != 200:
        raise RuntimeError(f"magic666 HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def _extract_text(data: dict[str, Any]) -> str:
    chunks: list[str] = []
    for choice in data.get("choices", []) or []:
        content = (choice.get("message", {}) or {}).get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            chunks.extend(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    text = "\n".join(c for c in chunks if c).strip()
    if not text:
        reasons = [c.get("finish_reason") for c in data.get("choices", []) or []]
        raise RuntimeError(f"empty completion (finish_reason={reasons})")
    return text


def _extract_image_urls(data: dict[str, Any]) -> list[str]:
    urls: list[str] = []
    for item in data.get("data", []) or []:
        if isinstance(item, dict):
            if item.get("url"):
                urls.append(str(item["url"]))
            elif item.get("b64_json"):
                urls.append("data:image/png;base64," + str(item["b64_json"]))
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
            urls.extend(re.findall(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+", content))
            urls.extend(re.findall(r"https?://[^\s)\"'<>]+\.(?:png|jpg|jpeg|webp|jfif)", content, re.I))
        for image in message.get("images", []) or []:
            if isinstance(image, str):
                urls.append(image)
            elif isinstance(image, dict):
                url = image.get("url") or (image.get("image_url") or {}).get("url")
                if url:
                    urls.append(str(url))
    return urls


def _download_image(url: str) -> Image.Image:
    if url.startswith("data:"):
        _, encoded = url.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _with_retries(label: str, fn: Callable[[], Any], retries: int = RETRIES) -> Any:
    last: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            return fn()
        except (RuntimeError, ValueError, KeyError, TypeError, requests.RequestException, OSError) as error:
            last = error
            print(f"[vlm] {label}: attempt {attempt}/{retries} failed: {str(error)[:300]}", flush=True)
            if attempt < retries:
                time.sleep(min(4 * attempt, 15))
    raise RuntimeError(f"{label} failed after {retries} attempts") from last


def parse_json(text: str) -> dict[str, Any]:
    # The gateway inserts stray raw newlines into completions, even inside JSON keys
    # ("instruction\n_following"). Raw newlines are never valid inside JSON strings (they would
    # be escaped as \n) and are optional between tokens, so dropping them restores the JSON.
    text = text.replace("\r", "").replace("\n", "")
    try:
        data = json.loads(text, strict=False)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"No JSON object in response: {text[:500]}")
    data = json.loads(match.group(0), strict=False)
    if not isinstance(data, dict):
        raise ValueError(f"JSON response is not an object: {text[:500]}")
    return data


def generate_json(
    prompt: str,
    images: list[ImageLike] | None,
    models: str | list[str],
    label: str,
    schema: dict[str, Any] | None = None,
    validate: Callable[[dict[str, Any]], None] | None = None,
    max_side: int | None = 768,
    retries: int = RETRIES,
    temperature: float | None = 0.0,
) -> dict[str, Any]:
    """Ask a VLM for a single JSON object; `validate` raises ValueError to trigger a retry.

    `models` may be a fallback list: each model gets `retries` attempts before the next is tried.
    """
    schema_text = f"\n\nJSON schema:\n{json.dumps(schema, ensure_ascii=False)}" if schema else ""
    full_prompt = f"{prompt}{schema_text}\n\nReturn only one valid JSON object. Do not include markdown."
    last: Exception | None = None
    for model in [models] if isinstance(models, str) else models:
        def once(model: str = model) -> dict[str, Any]:
            start = time.time()
            raw = _request(model, full_prompt, images, max_side, response_json=True, temperature=temperature)
            data = parse_json(_extract_text(raw))
            if validate is not None:
                try:
                    validate(data)
                except (ValueError, KeyError, TypeError) as error:
                    _log({"label": label, "model": model, "rejected": str(error), "response": data})
                    raise
            _log({"label": label, "model": model, "seconds": round(time.time() - start, 1),
                  "num_images": len(images or []), "prompt": full_prompt, "response": data})
            return data

        try:
            return _with_retries(f"{label} [{model}]", once, retries)
        except RuntimeError as error:
            last = error
            print(f"[vlm] {label}: model {model} exhausted, trying next fallback", flush=True)
    raise RuntimeError(f"{label}: all VLM models failed") from last


def generate_image(
    prompt: str,
    images: list[ImageLike],
    models: list[str],
    label: str,
    max_side: int | None = 1024,
    retries: int = RETRIES,
) -> tuple[Image.Image, str]:
    """Image editing with model fallback. Returns (image, model_used)."""
    last: Exception | None = None
    for model in models:
        def once(model: str = model) -> Image.Image:
            start = time.time()
            raw = _request(model, prompt, images, max_side)
            urls = _extract_image_urls(raw)
            if not urls:
                text = ""
                try:
                    text = _extract_text(raw)
                except RuntimeError:
                    pass
                raise RuntimeError(f"no image in response; text={text[:300]!r}")
            image = _download_image(urls[0])
            _log({"label": label, "model": model, "seconds": round(time.time() - start, 1),
                  "num_images": len(images), "prompt": prompt, "result_url": urls[0][:200]})
            return image

        try:
            return _with_retries(f"{label} [{model}]", once, retries), model
        except RuntimeError as error:
            last = error
            print(f"[vlm] {label}: model {model} exhausted, trying next fallback", flush=True)
    raise RuntimeError(f"{label}: all image models failed") from last
