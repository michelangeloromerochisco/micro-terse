"""OpenAI-compatible FastAPI server for Micro-Terse inference."""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware

DEFAULT_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]

MAX_CONTENT_LENGTH = 4096


def _validate_cors_origins(origins: list[str]) -> list[str]:
    """Reject wildcard and obviously invalid origins; default to localhost if none valid."""
    valid: list[str] = []
    for o in origins:
        o = o.strip()
        if o == "*":
            continue
        if not o.startswith(("http://", "https://")):
            continue
        valid.append(o)
    return valid or DEFAULT_ORIGINS

DEMO_RESPONSE = (
    "This is the Micro-Terse demo server running in placeholder mode. "
    "Load a checkpoint or GGUF to serve the real model.\n\n"
    "Key facts:\n"
    "- 423M parameters with ternary weights {-1, 0, +1}\n"
    "- 16× memory compression versus BF16 baselines\n"
    "- Pure-PyTorch training stack, no proprietary frameworks\n"
    "- Runs on consumer GPUs and edge hardware\n"
    "- First Colombian ternary-weight LLM"
)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(..., max_length=MAX_CONTENT_LENGTH)


class ChatCompletionRequest(BaseModel):
    model: str = "terse-micro"
    messages: list[ChatMessage]
    stream: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("messages")
    @classmethod
    def _limit_total_chars(cls, messages: list[ChatMessage]) -> list[ChatMessage]:
        total = sum(len(m.content) for m in messages)
        if total > 10 * MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Total prompt exceeds {10 * MAX_CONTENT_LENGTH} characters"
            )
        return messages


class PredictRequest(BaseModel):
    text: str = Field(..., max_length=MAX_CONTENT_LENGTH)
    k: int = Field(default=5, ge=1, le=10)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "terse"


def _default_state() -> dict[str, Any]:
    return {
        "model": None,
        # Optional second model used ONLY for /v1/identity_proof. Lets the demo
        # serve the more fluent SFT model for chat while still proving identity
        # alignment on the ORPO model. Falls back to `model` when unset.
        "proof_model": None,
        # Optional model used ONLY for /v1/predict* (next-token demo). The base
        # pretrained model is the strongest here. Falls back to `model`.
        "base_model": None,
        "tokenizer": None,
        "device": "cpu",
        "demo_mode": True,
        "demo_response": None,
        "cors_origins": DEFAULT_ORIGINS,
        "api_key": None,
        "identity_progression": None,
    }


def _last_user_text(messages: list[ChatMessage]) -> str:
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""


def _sse_chunk(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _stream_placeholder(text: str, request_id: str, delay: float = 0.02) -> StreamingResponse:
    """Yield a canned response word-by-word so the client exercises streaming."""
    words = text.split(" ")
    total_words = len(words)

    async def generator() -> AsyncGenerator[str, None]:
        for i, word in enumerate(words):
            payload = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "terse-micro",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": word + ("" if i == total_words - 1 else " ")},
                        "finish_reason": None,
                    }
                ],
            }
            yield _sse_chunk(payload)
            await _async_sleep(delay)

        yield _sse_chunk(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "terse-micro",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(generator(), media_type="text/event-stream")


def _encode_chat(model, tokenizer, messages: list[ChatMessage], device: str):
    from terse.model.generate import apply_chatml_template

    prompt = apply_chatml_template([m.model_dump() for m in messages])
    input_ids = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True)
    return input_ids.to(device), len(input_ids[0])


def _create_non_streaming_response(content: str, request: ChatCompletionRequest) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


class _APIKeyMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_key: str | None) -> None:
        super().__init__(app)
        self.api_key = api_key

    async def dispatch(self, request, call_next):
        if self.api_key and request.url.path.startswith("/v1/"):
            auth = request.headers.get("authorization", "")
            if not auth.startswith("Bearer ") or auth[7:] != self.api_key:
                return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)


def build_app(state: dict[str, Any]) -> FastAPI:
    """Create a fresh FastAPI application bound to the given server state."""
    app = FastAPI(title="Micro-Terse Server")

    @app.middleware("http")
    async def _security_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        return response

    origins = _validate_cors_origins(state["cors_origins"])
    app.add_middleware(_APIKeyMiddleware, api_key=state.get("api_key"))
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
        allow_credentials=False,
    )

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"object": "list", "data": [ModelInfo(id="terse-micro").model_dump()]}

    @app.get("/v1/status")
    def get_status() -> dict:
        return {
            "model": "terse-micro",
            "demo_mode": bool(state["demo_mode"]),
            "device": str(state["device"]),
        }

    @app.get("/v1/identity_proof")
    def identity_proof() -> dict:
        """Live, measurable proof of identity alignment on the loaded model.

        Scores each identity probe's charter answer vs. its "ChatGPT" answer and
        returns the per-probe margins. A positive margin means the model assigns
        higher probability to being Terse than to being ChatGPT.
        """
        # Prefer the dedicated proof model (ORPO) when one is loaded; otherwise
        # score whatever chat model is serving.
        proof_model = state.get("proof_model") or state["model"]
        if state["demo_mode"] or proof_model is None or state["tokenizer"] is None:
            return {"available": False, "probes": [], "preferred": 0, "total": 0}

        from terse.server.identity import identity_margins

        probes = identity_margins(proof_model, state["tokenizer"], state["device"])
        preferred = sum(1 for p in probes if p["prefers_charter"])
        response = {
            "available": True,
            "probes": probes,
            "preferred": preferred,
            "total": len(probes),
        }
        progression = state.get("identity_progression")
        if progression:
            response["progression"] = progression
        return response

    def _predict_model():
        # The base model is the strongest at next-token prediction; fall back to
        # whatever chat model is loaded.
        return state.get("base_model") or state["model"]

    @app.post("/v1/predict")
    def predict(request: PredictRequest) -> dict:
        """Top-k next-token prediction — the base model's headline capability."""
        model = _predict_model()
        if state["demo_mode"] or model is None or state["tokenizer"] is None:
            return {"available": False, "predictions": []}

        from terse.server.predict import next_token_topk

        preds = next_token_topk(model, state["tokenizer"], state["device"], request.text, request.k)
        return {
            "available": True,
            "text": request.text,
            "predictions": preds,
            "model": "base" if state.get("base_model") else "chat",
        }

    @app.get("/v1/predict_showcase")
    def predict_showcase() -> dict:
        """Curated next-token prompts the base model reliably nails."""
        model = _predict_model()
        if state["demo_mode"] or model is None or state["tokenizer"] is None:
            return {"available": False, "items": []}

        from terse.server.predict import showcase

        return {
            "available": True,
            "items": showcase(model, state["tokenizer"], state["device"]),
            "model": "base" if state.get("base_model") else "chat",
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: ChatCompletionRequest, raw_request: Request
    ):
        request_id = f"chatcmpl-{uuid.uuid4().hex}"

        if state["demo_mode"]:
            content = state.get("demo_response") or DEMO_RESPONSE
            user_text = _last_user_text(request.messages).strip()
            if user_text:
                content = f"Demo mode: you asked about *{user_text[:80]}*.\n\n{content}"
            if not request.stream:
                return _create_non_streaming_response(content, request)
            return _stream_placeholder(content, request_id)

        model = state["model"]
        tokenizer = state["tokenizer"]
        if model is None or tokenizer is None:
            return _create_non_streaming_response(
                "Error: model not loaded and demo mode is disabled.", request
            )

        device = state["device"]
        input_ids, prompt_len = _encode_chat(model, tokenizer, request.messages, device)
        # ChatML turn markers aren't special tokens here, so the model signals
        # end-of-turn with the literal string, not eos. Stop on the decoded text.
        stop_texts = ("<|im_end|>", "<|im_start|>")

        if not request.stream:
            from terse.model.generate import decode_with_reasoning, generate_stream

            try:
                # generate_stream rebuilds the id tensor each step (torch.cat
                # returns a new tensor), so the prompt-only `input_ids` we hold
                # never grows — decode the last yielded full ids, not `input_ids`.
                final_ids = input_ids
                for _, full_ids in generate_stream(
                    model,
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    min_new_tokens=min(16, request.max_tokens),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_texts=stop_texts,
                    tokenizer=tokenizer,
                    device=device,
                ):
                    final_ids = full_ids
                content = decode_with_reasoning(tokenizer, final_ids, prompt_len)
            except Exception:
                content = (
                    "Generation failed. Please try a shorter prompt or check the server logs."
                )
            return _create_non_streaming_response(content, request)

        async def stream_real() -> AsyncGenerator[str, None]:
            from terse.model.generate import decode_with_reasoning, generate_stream

            previous_text = ""
            eos_token_id = tokenizer.eos_token_id

            try:
                for token_id, full_ids in generate_stream(
                    model,
                    input_ids,
                    max_new_tokens=request.max_tokens,
                    min_new_tokens=min(16, request.max_tokens),
                    temperature=request.temperature,
                    top_p=request.top_p,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    stop_texts=stop_texts,
                    tokenizer=tokenizer,
                    device=device,
                ):
                    current_text = decode_with_reasoning(tokenizer, full_ids, prompt_len)
                    delta = current_text[len(previous_text):]
                    previous_text = current_text

                    if delta:
                        yield _sse_chunk(
                            {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )

                    if token_id == eos_token_id:
                        break
            except Exception:
                yield _sse_chunk(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": "\n[Generation failed. Check server logs.]"
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )

            yield _sse_chunk(
                {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_real(), media_type="text/event-stream")

    return app


def create_app(
    model=None,
    tokenizer=None,
    device: str = "cpu",
    demo_mode: bool = True,
    demo_response: str | None = None,
    cors_origins: list[str] | None = None,
    api_key: str | None = None,
    identity_progression: list | None = None,
    proof_model=None,
    base_model=None,
) -> FastAPI:
    """Create a new server instance with the given runtime configuration."""
    state = _default_state()
    state["model"] = model
    state["proof_model"] = proof_model
    state["base_model"] = base_model
    state["tokenizer"] = tokenizer
    state["device"] = device
    state["demo_mode"] = demo_mode
    state["demo_response"] = demo_response
    state["cors_origins"] = _validate_cors_origins(cors_origins or DEFAULT_ORIGINS)
    state["api_key"] = api_key
    state["identity_progression"] = identity_progression
    return build_app(state)


# Default app instance used when importing the module (e.g. by uvicorn).
app = create_app()
