"""OpenAI Codex Responses Provider.

Mapped from the TypeScript reference implementation at:
https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-codex-responses.ts
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncGenerator, Literal

import httpx
from loguru import logger
from oauth_cli_kit import get_token as get_codex_token

from nanobot.providers.base import LLMProvider, LLMResponse, ToolCallRequest

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
DEFAULT_ORIGINATOR = "nanobot"

MAX_RETRIES = 3
BASE_DELAY_MS = 1000

# Effort levels — mapped directly from the TS reference.
# "xhigh" is the TS name; we also accept "extra_high" as a Python-friendly alias.
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "extra_high"]
ReasoningSummary = Literal["auto", "concise", "detailed", "off", "on"]
TextVerbosity = Literal["low", "medium", "high"]

_CODEX_RESPONSE_STATUSES = frozenset(
    {"completed", "incomplete", "failed", "cancelled", "queued", "in_progress"}
)


# ============================================================================
# Per-model effort clamping (matches TS clampReasoningEffort)
# ============================================================================

def _clamp_reasoning_effort(model_id: str, effort: str) -> str:
    """Apply model-specific caps on reasoning effort.

    Mirrors the TypeScript ``clampReasoningEffort`` function exactly.
    """
    # Strip provider prefix (e.g. "openai-codex/gpt-5.1-codex" → "gpt-5.1-codex")
    mid = model_id.split("/")[-1] if "/" in model_id else model_id

    # Normalise our Python alias to the API value
    if effort == "extra_high":
        effort = "xhigh"

    if (mid.startswith("gpt-5.2") or mid.startswith("gpt-5.3")) and effort == "minimal":
        return "low"
    if mid == "gpt-5.1" and effort == "xhigh":
        return "high"
    if mid == "gpt-5.1-codex-mini":
        return "high" if effort in ("high", "xhigh") else "medium"

    return effort


# ============================================================================
# Provider class
# ============================================================================

class OpenAICodexProvider(LLMProvider):
    """Use Codex OAuth to call the Responses API."""

    def __init__(
        self,
        default_model: str = "openai-codex/gpt-5.1-codex",
        reasoning_effort: ReasoningEffort = "medium",
        reasoning_summary: ReasoningSummary = "auto",
        text_verbosity: TextVerbosity = "medium",
    ):
        super().__init__(api_key=None, api_base=None)
        self.default_model = default_model
        self.reasoning_effort = reasoning_effort
        self.reasoning_summary = reasoning_summary
        self.text_verbosity = text_verbosity

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        model = model or self.default_model
        effort = reasoning_effort or self.reasoning_effort
        summary = reasoning_summary or self.reasoning_summary

        system_prompt, input_items = _convert_messages(messages)

        token = await asyncio.to_thread(get_codex_token)
        headers = _build_headers(token.account_id, token.access)

        body: dict[str, Any] = {
            "model": _strip_model_prefix(model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": self.text_verbosity},
            "include": ["reasoning.encrypted_content"],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }

        # Reasoning config
        if effort and effort != "none":
            body["reasoning"] = {
                "effort": _clamp_reasoning_effort(model, effort),
                "summary": summary,
            }

        if tools:
            body["tools"] = _convert_tools(tools)

        logger.info(
            "Codex ▶ model={} effort={} summary={} items={} tools={}",
            _strip_model_prefix(model),
            effort,
            summary,
            len(input_items),
            len(body.get("tools") or []),
        )
        logger.debug("Codex request body: {}", json.dumps(
            {k: v for k, v in body.items() if k not in ("input", "instructions")},
            ensure_ascii=False,
        ))

        url = _resolve_codex_url()

        try:
            content, reasoning_content, tool_calls, finish_reason = await _request_codex_with_retry(
                url, headers, body
            )
            logger.info(
                "Codex ◀ finish={} content={}ch reasoning={}ch tool_calls={}",
                finish_reason,
                len(content or ""),
                len(reasoning_content or ""),
                len(tool_calls),
            )
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                reasoning_content=reasoning_content or None,
            )
        except Exception as e:
            return LLMResponse(
                content=f"Error calling Codex: {str(e)}",
                finish_reason="error",
            )

    def get_default_model(self) -> str:
        return self.default_model


# ============================================================================
# URL helpers
# ============================================================================

def _resolve_codex_url(base_url: str = "") -> str:
    """Build the full codex/responses endpoint URL (matches TS resolveCodexUrl)."""
    raw = base_url.strip() if base_url else DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def _strip_model_prefix(model: str) -> str:
    if model.startswith("openai-codex/") or model.startswith("openai_codex/"):
        return model.split("/", 1)[1]
    return model


# ============================================================================
# Headers
# ============================================================================

def _build_headers(account_id: str, token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "nanobot (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


# ============================================================================
# Retry logic (matches TS retry behaviour)
# ============================================================================

def _is_retryable(status: int, body: str) -> bool:
    if status in (429, 500, 502, 503, 504):
        return True
    return bool(
        re.search(
            r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
            body,
            re.IGNORECASE,
        )
    )


async def _request_codex_with_retry(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
) -> tuple[str, str, list[ToolCallRequest], str]:
    """Fetch with exponential-backoff retry on rate limits / transient errors.

    Returns (content, reasoning_content, tool_calls, finish_reason).
    """
    last_error: Exception | None = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            result = await _request_codex(url, headers, body, verify=True)
            return result
        except _RetryableError as e:
            last_error = e
            delay = BASE_DELAY_MS * (2 ** attempt) / 1000
            logger.warning(
                "Codex request failed (attempt {}/{}): {} — retrying in {:.1f}s",
                attempt + 1, MAX_RETRIES + 1, e, delay,
            )
            await asyncio.sleep(delay)
        except _SslError:
            logger.warning("SSL certificate verification failed for Codex API; retrying with verify=False")
            return await _request_codex(url, headers, body, verify=False)
        except Exception:
            raise

    raise last_error or RuntimeError("Codex request failed after retries")


class _RetryableError(Exception):
    """Signals a transient error that should be retried."""


class _SslError(Exception):
    """Signals an SSL verification failure."""


async def _request_codex(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    verify: bool = True,
) -> tuple[str, str, list[ToolCallRequest], str]:
    """Single attempt. Raises _RetryableError, _SslError, or RuntimeError."""
    try:
        async with httpx.AsyncClient(timeout=300.0, verify=verify) as client:
            logger.debug("Codex HTTP POST {} (verify={})", url, verify)
            async with client.stream("POST", url, headers=headers, json=body) as response:
                logger.debug("Codex HTTP response status={}", response.status_code)
                if response.status_code != 200:
                    raw = (await response.aread()).decode("utf-8", "ignore")
                    msg = _parse_error_response(response.status_code, raw)
                    if _is_retryable(response.status_code, raw):
                        raise _RetryableError(msg)
                    raise RuntimeError(msg)
                return await _consume_sse(response)
    except httpx.ConnectError as e:
        raise _RetryableError(str(e)) from e
    except Exception as e:
        if "CERTIFICATE_VERIFY_FAILED" in str(e):
            raise _SslError(str(e)) from e
        raise


# ============================================================================
# Tool schema conversion
# ============================================================================

def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI function-calling schema to Codex flat format."""
    converted: list[dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append({
            "type": "function",
            "name": name,
            "description": fn.get("description") or "",
            "parameters": params if isinstance(params, dict) else {},
        })
    return converted


# ============================================================================
# Message conversion
# ============================================================================

def _convert_messages(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    system_prompt = ""
    input_items: list[dict[str, Any]] = []

    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")

        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue

        if role == "user":
            input_items.append(_convert_user_message(content))
            continue

        if role == "assistant":
            # Handle text first.
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            # Then handle tool calls.
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                call_id = call_id or f"call_{idx}"
                item_id = item_id or f"fc_{idx}"
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id,
                        "call_id": call_id,
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue

        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            output_text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
            continue

    return system_prompt, input_items


def _convert_user_message(content: Any) -> dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append({"type": "input_image", "image_url": url, "detail": "auto"})
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> tuple[str, str | None]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


# ============================================================================
# SSE parsing
# ============================================================================

async def _iter_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    buffer: list[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [l[5:].strip() for l in buffer if l.startswith("data:")]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(response: httpx.Response) -> tuple[str, str, list[ToolCallRequest], str]:
    """Consume SSE stream. Returns (content, reasoning_content, tool_calls, finish_reason)."""
    content = ""
    reasoning_content = ""
    tool_calls: list[ToolCallRequest] = []
    tool_call_buffers: dict[str, dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in _iter_sse(response):
        event_type = event.get("type")

        # Log every event type at DEBUG; skip high-frequency delta events at that level
        if event_type and not event_type.endswith(".delta"):
            logger.debug("Codex SSE event: {}", event_type)

        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                logger.debug("Codex tool call queued: name={} call_id={}", item.get("name"), call_id)
                tool_call_buffers[call_id] = {
                    "id": item.get("id") or "fc_0",
                    "name": item.get("name"),
                    "arguments": item.get("arguments") or "",
                }

        elif event_type == "response.output_text.delta":
            content += event.get("delta") or ""

        # Reasoning text — streamed as deltas then finalised
        elif event_type == "response.reasoning.delta":
            reasoning_content += event.get("delta") or ""
        elif event_type == "response.reasoning.done":
            if text := event.get("text"):
                reasoning_content = text  # authoritative final value
            logger.debug("Codex reasoning done: {}ch", len(reasoning_content))

        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""

        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""

        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"raw": args_raw}
                tc = ToolCallRequest(
                    id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                    name=buf.get("name") or item.get("name"),
                    arguments=args,
                )
                logger.debug("Codex tool call ready: name={} call_id={}", tc.name, call_id)
                tool_calls.append(tc)

        # "response.done" is an alias used by some Codex versions (matches TS mapCodexEvents)
        elif event_type in ("response.completed", "response.done"):
            resp = event.get("response") or {}
            status = _normalize_codex_status(resp.get("status"))
            finish_reason = _map_finish_reason(status)
            logger.debug("Codex response {} status={}", event_type, status)
            # Fallback: extract reasoning from the output array in the completed response
            if not reasoning_content:
                reasoning_content = _extract_reasoning_from_response(resp)

        elif event_type == "response.failed":
            # Extract the error message from the response object (matches TS mapCodexEvents)
            msg = ((event.get("response") or {}).get("error") or {}).get("message")
            raise RuntimeError(msg or "Codex response failed")

        elif event_type == "error":
            code = event.get("code") or ""
            raise RuntimeError(f"Codex error: {msg or code or json.dumps(event)}")

    # Clean up any bleeding raw prompt tokens or strange language artifacts 
    # that Codex occasionally leaks into the text stream during function calls.
    if content:
        content = _strip_special_tokens(content)

    return content, reasoning_content, tool_calls, finish_reason


# ============================================================================
# Response helpers
# ============================================================================

def _strip_special_tokens(text: str) -> str:
    """Remove common Codex literal prompt tokens leaked into visible text."""
    # Remove `+assistant to=functions...` syntax
    text = re.sub(r'(\+assistant.*?(?:to=|functions\.).*?(?:\n|$))', '', text, flags=re.IGNORECASE)
    # Remove large unescaped JSON bloat if it's clearly a raw leaked tool call dict
    text = re.sub(r'({"command":\s*".*?"}\s*)$', '', text)
    # Remove specific artifact you reported: +assistant to=functions.exec...
    text = re.sub(r'\+assistant[\s\S]*?(?:exec|մեկնաբանություն|大发快三计划)[\s\S]*?(?=\n\n|\Z)', '', text)
    # Remove any trailing invisible characters or stray JSON braces
    return text.strip()

def _normalize_codex_status(status: Any) -> str | None:
    """Normalise Codex status field to a known value (matches TS normalizeCodexStatus)."""
    if isinstance(status, str) and status in _CODEX_RESPONSE_STATUSES:
        return status
    return None


def _extract_reasoning_from_response(response: dict[str, Any]) -> str:
    """Extract plaintext reasoning from a completed response object (fallback).

    The Codex API embeds reasoning inside the output array as an item with
    type 'reasoning' whose 'content' is a list of text blocks.
    """
    for item in (response.get("output") or []):
        if item.get("type") == "reasoning":
            parts = [
                block.get("text") or ""
                for block in (item.get("content") or [])
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            if parts:
                return "".join(parts)
    return ""


_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
    "queued": "stop",
    "in_progress": "stop",
}


def _map_finish_reason(status: str | None) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


# ============================================================================
# Error parsing (matches TS parseErrorResponse)
# ============================================================================

def _parse_error_response(status_code: int, raw: str) -> str:
    """Build a user-friendly error message from an API error response body."""
    friendly: str | None = None
    try:
        parsed = json.loads(raw)
        err = parsed.get("error") or {}
        code = err.get("code") or err.get("type") or ""
        if re.search(r"usage_limit_reached|usage_not_included|rate_limit_exceeded", code, re.I) or status_code == 429:
            plan = f" ({err['plan_type'].lower()} plan)" if err.get("plan_type") else ""
            resets_at = err.get("resets_at")
            if resets_at:
                import time
                mins = max(0, round((resets_at * 1000 - time.time() * 1000) / 60000))
                when = f" Try again in ~{mins} min."
            else:
                when = ""
            friendly = f"You have hit your ChatGPT usage limit{plan}.{when}".strip()
        msg = err.get("message") or friendly or raw
        return msg
    except Exception:
        pass
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limited. Please try again later."
    return f"HTTP {status_code}: {raw}"
