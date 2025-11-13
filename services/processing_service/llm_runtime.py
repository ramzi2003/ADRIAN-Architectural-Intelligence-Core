"""
Local LLM runtime wrapper for ADRIAN Processing Service.

Provides a coroutine-friendly API around llama.cpp (GGUF) models with support
for streaming token generation, warmup routines, and health reporting.
"""
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Optional

from llama_cpp import Llama

from shared.config import Settings, get_settings
from shared.logging_config import setup_logging


logger = setup_logging("processing-service.llm-runtime")


@dataclass
class LLMHealth:
    """Health status for the LLM runtime."""

    loaded: bool
    model_path: str
    model_size_bytes: Optional[int]
    last_error: Optional[str] = None


class LLMRuntimeError(RuntimeError):
    """Custom exception for runtime failures."""


class LLMRuntime:
    """
    Thin async wrapper over llama.cpp bindings.

    Example:
        runtime = LLMRuntime()
        await runtime.ensure_loaded()
        text = await runtime.generate("Explain ADRIAN.")
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self._settings = settings or get_settings()
        self._model_path = Path(self._settings.llm_model_path).expanduser()
        self._model: Optional[Llama] = None
        self._model_lock = asyncio.Lock()
        self._load_lock = asyncio.Lock()
        self._last_error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    async def ensure_loaded(self) -> None:
        """Load the model into memory if it is not already available."""
        if self.is_loaded:
            return

        async with self._load_lock:
            if self.is_loaded:
                return

            if not self._model_path.exists():
                self._last_error = f"Model file not found at {self._model_path}"
                raise LLMRuntimeError(self._last_error)

            logger.info(
                "Loading GGUF model from %s (threads=%s, context=%s, gpu_layers=%s)",
                self._model_path,
                self._settings.llm_thread_count,
                self._settings.llm_context_window,
                self._settings.llm_gpu_layers,
            )

            loop = asyncio.get_running_loop()
            try:
                self._model = await loop.run_in_executor(
                    None,
                    lambda: Llama(
                        model_path=str(self._model_path),
                        n_ctx=self._settings.llm_context_window,
                        n_threads=self._settings.llm_thread_count,
                        n_gpu_layers=self._settings.llm_gpu_layers,
                        last_n_tokens_size=256,
                        vocab_only=False,
                        use_mlock=False,
                        embedding=False,
                    ),
                )
                self._last_error = None
                logger.info("LLM model loaded successfully.")
            except Exception as exc:  # pragma: no cover - runtime specific
                self._last_error = str(exc)
                logger.exception("Failed to load GGUF model: %s", exc)
                raise LLMRuntimeError(self._last_error) from exc

    async def generate(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate a completion for the given prompt."""
        model = await self._get_model()

        params = self._build_sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

        loop = asyncio.get_running_loop()
        try:
            completion = await loop.run_in_executor(
                None,
                lambda: model.create_completion(
                    prompt=prompt, stream=False, **params
                ),
            )
            return completion["choices"][0]["text"]
        except Exception as exc:  # pragma: no cover
            self._last_error = str(exc)
            logger.exception("LLM generation failed: %s", exc)
            raise LLMRuntimeError(self._last_error) from exc

    async def stream(
        self,
        prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repeat_penalty: Optional[float] = None,
        stop: Optional[list[str]] = None,
    ) -> AsyncIterator[str]:
        """Stream tokens asynchronously for a prompt."""
        model = await self._get_model()
        params = self._build_sampling_params(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            stop=stop,
        )

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def _run_stream() -> None:
            try:
                for chunk in model.create_completion(
                    prompt=prompt, stream=True, **params
                ):
                    token = chunk["choices"][0]["text"]
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, token)
                loop.call_soon_threadsafe(queue.put_nowait, None)
            except Exception as exc:  # pragma: no cover
                logger.exception("LLM streaming failed: %s", exc)
                self._last_error = str(exc)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_run_stream, daemon=True).start()

        while True:
            token = await queue.get()
            if token is None:
                break
            yield token

    async def warmup(self) -> None:
        """Run a short prompt to warm the model (optional)."""
        if not self._settings.llm_warmup_prompt:
            return

        try:
            await self.generate(
                self._settings.llm_warmup_prompt,
                temperature=0.7,
                max_tokens=min(64, self._settings.llm_max_output_tokens),
            )
            logger.info("LLM warmup completed.")
        except LLMRuntimeError as exc:
            logger.warning("LLM warmup failed: %s", exc)

    async def health(self) -> LLMHealth:
        """Return health information for diagnostics."""
        try:
            stats = self._model_path.stat().st_size
        except FileNotFoundError:
            stats = None

        return LLMHealth(
            loaded=self.is_loaded,
            model_path=str(self._model_path),
            model_size_bytes=stats,
            last_error=self._last_error,
        )

    def _build_sampling_params(
        self,
        *,
        temperature: Optional[float],
        max_tokens: Optional[int],
        top_p: Optional[float],
        top_k: Optional[int],
        repeat_penalty: Optional[float],
        stop: Optional[list[str]],
    ) -> dict:
        """Merge custom overrides with defaults from settings."""
        return {
            "temperature": temperature or self._settings.llm_temperature,
            "max_tokens": max_tokens or self._settings.llm_max_output_tokens,
            "top_p": top_p or self._settings.llm_top_p,
            "top_k": top_k or self._settings.llm_top_k,
            "repeat_penalty": repeat_penalty or self._settings.llm_repeat_penalty,
            "stop": stop or [],
        }

    async def _get_model(self) -> Llama:
        """Ensure the model is loaded and return it."""
        await self.ensure_loaded()
        assert self._model is not None  # for type checking
        return self._model


# Module-level singleton helper ------------------------------------------------
_runtime_instance: Optional[LLMRuntime] = None
_runtime_lock: Optional[asyncio.Lock] = None


async def get_runtime() -> LLMRuntime:
    """Get a cached runtime instance."""
    global _runtime_instance
    if _runtime_instance is not None:
        return _runtime_instance
    global _runtime_lock
    if _runtime_lock is None:
        _runtime_lock = asyncio.Lock()
    async with _runtime_lock:
        if _runtime_instance is None:
            _runtime_instance = LLMRuntime()
            await _runtime_instance.ensure_loaded()
        return _runtime_instance

