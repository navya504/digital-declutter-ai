"""OpenAI-compatible inference against Nebius, Hugging Face TGI, or Ollama."""
from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

PROVIDER = os.getenv("DECLUTTER_PROVIDER", "nebius").strip().lower()
BASE_MODEL = os.getenv("DECLUTTER_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
FINE_TUNED_MODEL = os.getenv("DECLUTTER_FINE_TUNED_MODEL", "")
NEBIUS_URL = os.getenv("NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
HF_ENDPOINT_URL = os.getenv("HF_ENDPOINT_URL", "").strip().rstrip("/")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_ADAPTER_ID = os.getenv("HF_ADAPTER_ID", FINE_TUNED_MODEL)
ALLOWED = {"KEEP", "DELETE", "ARCHIVE", "REVIEW", "UNSUBSCRIBE"}
logger = logging.getLogger("uvicorn.error")


def _read_prediction(text: str) -> dict:
    match = re.search(r"\{.*?\}", text, flags=re.S)
    if match:
        try:
            data = json.loads(match.group(0))
            action = str(data.get("action", "REVIEW")).upper()
            return {"action": action if action in ALLOWED else "REVIEW", "reason": str(data.get("reason", "")), "confidence": _confidence(data.get("confidence"))}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    action_match = re.search(r"\b(KEEP|DELETE|ARCHIVE|REVIEW|UNSUBSCRIBE)\b", text.upper())
    return {"action": action_match.group(1) if action_match else "REVIEW", "reason": text.strip()[:300] or "The model did not return a usable explanation; review this item.", "confidence": None}


def _confidence(value):
    try:
        number = float(value)
        return round(number, 4) if 0 <= number <= 1 else None
    except (TypeError, ValueError):
        return None


class ModelService:
    def __init__(self):
        self.provider = PROVIDER
        if self.provider == "nebius":
            self.base_url = NEBIUS_URL
            self.api_key = os.getenv("NEBIUS_API_KEY", "")
        elif self.provider == "ollama":
            self.base_url = OLLAMA_URL
            self.api_key = "ollama"
        elif self.provider == "huggingface":
            if HF_ENDPOINT_URL and not HF_ENDPOINT_URL.endswith("/v1"):
                self.base_url = f"{HF_ENDPOINT_URL}/v1"
            else:
                self.base_url = HF_ENDPOINT_URL
            self.api_key = HF_TOKEN
        else:
            raise RuntimeError("DECLUTTER_PROVIDER must be 'nebius', 'huggingface', or 'ollama'.")

    @property
    def configured(self) -> bool:
        if not FINE_TUNED_MODEL:
            return False
        if self.provider == "ollama":
            return True
        if self.provider == "huggingface":
            return bool(HF_ENDPOINT_URL and HF_TOKEN and HF_ADAPTER_ID)
        return bool(self.api_key)

    @property
    def configuration_message(self) -> str:
        if self.provider == "nebius" and not self.api_key:
            return "Set NEBIUS_API_KEY in the backend environment. Keep it out of frontend code and source control."
        if self.provider == "huggingface" and not HF_ENDPOINT_URL:
            return "Set HF_ENDPOINT_URL to the deployed Hugging Face TGI endpoint URL."
        if self.provider == "huggingface" and not HF_TOKEN:
            return "Set HF_TOKEN in the backend environment. Keep it out of frontend code and source control."
        if self.provider == "huggingface" and not HF_ADAPTER_ID:
            return "Set HF_ADAPTER_ID to the LoRA adapter repository loaded by the endpoint."
        if not FINE_TUNED_MODEL:
            return "Set DECLUTTER_FINE_TUNED_MODEL to the deployed LoRA model name/repository or installed local model tag."
        return "Configured"

    def unload(self) -> None:
        # Hosted endpoints and Ollama keep model weights outside this process.
        return None

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Missing OpenAI-compatible client. Install the project requirements first.") from exc
        return OpenAI(base_url=self.base_url, api_key=self.api_key or "ollama", timeout=180)

    def _predict_one(self, client, model: str, row: dict) -> dict:
        messages = [
            {"role": "system", "content": "You are a cautious digital organization classifier. Return ONLY one JSON object with action (one of KEEP, DELETE, ARCHIVE, REVIEW, UNSUBSCRIBE), confidence from 0 to 1, and a short reason. Choose REVIEW when evidence is insufficient. Never claim an item was deleted."},
            {"role": "user", "content": f"Digital asset domain: {row.get('domain', 'unknown')}\nItem details: {row.get('item_text') or row.get('text')}"},
        ]
        response = client.chat.completions.create(model=model, messages=messages, temperature=0, max_tokens=220)
        return _read_prediction(response.choices[0].message.content or "")

    def predict_batch(self, rows: list[dict], fine_tuned: bool) -> list[dict]:
        model = FINE_TUNED_MODEL if fine_tuned else BASE_MODEL
        if not model:
            raise FileNotFoundError("Fine-tuned model is not configured. Set DECLUTTER_FINE_TUNED_MODEL after deploying the adapter.")
        if self.provider in {"nebius", "huggingface"} and not self.api_key:
            key_name = "HF_TOKEN" if self.provider == "huggingface" else "NEBIUS_API_KEY"
            raise RuntimeError(f"{key_name} is not configured for the backend.")
        client = self._client()
        # TGI maps `model="tgi"` to the base model and any other model value to a
        # preloaded LoRA adapter ID. Keep the user-facing model names in health/UI.
        api_model = ("tgi" if not fine_tuned else HF_ADAPTER_ID) if self.provider == "huggingface" else model
        phase = "fine-tuned" if fine_tuned else "base"
        total = len(rows)
        started = time.monotonic()
        predictions = []
        logger.info("Analysis %s pass started: 0/%d rows | model=%s", phase, total, model)
        for index, row in enumerate(rows, start=1):
            try:
                predictions.append(self._predict_one(client, api_model, row))
            except Exception:
                logger.exception("Analysis %s pass failed at row %d/%d | model=%s", phase, index, total, model)
                raise
            elapsed = time.monotonic() - started
            average_per_row = elapsed / index
            eta_seconds = average_per_row * (total - index)
            logger.info(
                "Analysis %s progress: %d/%d rows (%.0f%%) | elapsed %.1fs | estimated remaining %.1fs",
                phase,
                index,
                total,
                index * 100 / total,
                elapsed,
                eta_seconds,
            )
        logger.info("Analysis %s pass complete: %d/%d rows | elapsed %.1fs", phase, total, total, time.monotonic() - started)
        return predictions


service = ModelService()
