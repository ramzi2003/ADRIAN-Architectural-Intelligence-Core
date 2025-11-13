"""
Intent classification module for ADRIAN Processing Service.

Provides a lightweight transformer-backed classifier with graceful fallback
to heuristic rules when the fine-tuned model is unavailable.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shared.config import Settings, get_settings
from shared.logging_config import setup_logging


logger = setup_logging("processing-service.intent-classifier")


INTENT_LABELS_DEFAULT: Sequence[str] = (
    "system_control",
    "search",
    "task_management",
    "conversation",
)


@dataclass
class IntentPrediction:
    label: str
    confidence: float
    parameters: dict
    used_classifier: bool


class IntentClassifierError(RuntimeError):
    """Raised when the classifier cannot be initialized or used."""


class IntentClassifier:
    """Async-friendly wrapper around a local HuggingFace classifier checkpoint."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model_path = Path(self._settings.intent_classifier_model_path).expanduser()
        self._device = torch.device(self._settings.intent_classifier_device)
        self._label_map = (
            list(self._settings.intent_classifier_label_map)
            if self._settings.intent_classifier_label_map
            else list(INTENT_LABELS_DEFAULT)
        )
        self._tokenizer: Optional[AutoTokenizer] = None
        self._model: Optional[AutoModelForSequenceClassification] = None
        self._load_lock = asyncio.Lock()
        self._last_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    async def ensure_loaded(self) -> None:
        if self.is_loaded:
            return

        async with self._load_lock:
            if self.is_loaded:
                return

            if not self._model_path.exists():
                self._last_error = f"Intent classifier model path missing: {self._model_path}"
                raise IntentClassifierError(self._last_error)

            logger.info(
                "Loading intent classifier from %s (labels=%s, device=%s)",
                self._model_path,
                self._label_map,
                self._device,
            )

            loop = asyncio.get_running_loop()
            try:
                self._tokenizer = await loop.run_in_executor(
                    None, lambda: AutoTokenizer.from_pretrained(self._model_path)
                )
                model = await loop.run_in_executor(
                    None, lambda: AutoModelForSequenceClassification.from_pretrained(self._model_path)
                )
                self._model = model.to(self._device)
                self._model.eval()
                self._last_error = None
                logger.info("Intent classifier loaded successfully.")
            except Exception as exc:  # pragma: no cover - heavy runtime
                self._last_error = str(exc)
                logger.exception("Failed to load intent classifier: %s", exc)
                raise IntentClassifierError(self._last_error) from exc

    async def predict(self, text: str, context: Optional[dict] = None) -> IntentPrediction:
        """
        Predict the intent label for the given text.

        Falls back to heuristics if the classifier is unavailable.
        """
        try:
            model = await self._get_model()
            tokenizer = self._tokenizer
            assert tokenizer is not None

            inputs = await self._encode(text, context, tokenizer)

            loop = asyncio.get_running_loop()
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None, lambda: model(**inputs).logits
                )

            probabilities = torch.softmax(outputs[0], dim=0)
            confidence, label_idx = torch.max(probabilities, dim=0)
            label = self._label_map[label_idx.item()] if label_idx.item() < len(self._label_map) else "conversation"
            return IntentPrediction(
                label=label,
                confidence=float(confidence.item()),
                parameters=_extract_parameters(label, text),
                used_classifier=True,
            )
        except IntentClassifierError:
            logger.warning(
                "Intent classifier unavailable, falling back to heuristic rules."
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Intent classifier prediction failed: %s", exc)

        heuristic_label, parameters = _heuristic_intent(text)
        return IntentPrediction(
            label=heuristic_label,
            confidence=0.5,
            parameters=parameters,
            used_classifier=False,
        )

    async def health(self) -> dict:
        """Return diagnostic info for health checks."""
        return {
            "model_path": str(self._model_path),
            "loaded": self.is_loaded,
            "device": str(self._device),
            "labels": list(self._label_map),
            "last_error": self._last_error,
        }

    async def _get_model(self) -> AutoModelForSequenceClassification:
        if not self.is_loaded:
            await self.ensure_loaded()
        assert self._model is not None
        return self._model

    async def _encode(
        self,
        text: str,
        context: Optional[dict],
        tokenizer: AutoTokenizer,
    ):
        """Tokenize text (optionally including conversation context)."""
        if self._settings.intent_classifier_enable_context and context:
            history = context.get("history") or []
            context_snippet = " ".join(turn.get("text", "") for turn in history[-2:])
            combined = f"{context_snippet}\n\nUser: {text}" if context_snippet else text
        else:
            combined = text

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: tokenizer(
                combined,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            ).to(self._device),
        )


def _heuristic_intent(text: str) -> tuple[str, dict]:
    """Fallback rule-based classification."""
    text_lower = text.lower()
    if "open" in text_lower:
        app_name = text_lower.replace("open", "").strip()
        return "system_control", {"action": "open", "target": app_name}
    if "close" in text_lower:
        app_name = text_lower.replace("close", "").strip()
        return "system_control", {"action": "close", "target": app_name}
    if any(keyword in text_lower for keyword in ("search", "find", "lookup")):
        return "search", {"query": text}
    if any(keyword in text_lower for keyword in ("remind", "todo", "task", "schedule")):
        return "task_management", {"text": text}
    return "conversation", {}


def _extract_parameters(label: str, text: str) -> dict:
    """Extract structured parameters from text based on predicted intent."""
    if label == "system_control":
        text_lower = text.lower()
        if "open" in text_lower:
            return {"action": "open", "target": text_lower.replace("open", "").strip()}
        if "close" in text_lower:
            return {"action": "close", "target": text_lower.replace("close", "").strip()}
    if label == "search":
        return {"query": text}
    if label == "task_management":
        return {"text": text}
    return {}


# Singleton helper ------------------------------------------------------------
_classifier_instance: Optional[IntentClassifier] = None
_classifier_lock: Optional[asyncio.Lock] = None


async def get_classifier() -> IntentClassifier:
    """Get a cached classifier instance, loading it on first access."""
    global _classifier_instance
    if _classifier_instance is not None:
        return _classifier_instance

    global _classifier_lock
    if _classifier_lock is None:
        _classifier_lock = asyncio.Lock()

    async with _classifier_lock:
        if _classifier_instance is None:
            _classifier_instance = IntentClassifier()
            try:
                await _classifier_instance.ensure_loaded()
            except IntentClassifierError as exc:
                logger.error("Intent classifier load failed: %s", exc)
            except Exception as exc:  # pragma: no cover
                logger.exception("Unexpected classifier init error: %s", exc)
        return _classifier_instance

