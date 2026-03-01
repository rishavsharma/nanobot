# nanobot Codebase Guide

This file is loaded into the agent's system prompt on every turn (see `ContextBuilder.BOOTSTRAP_FILES`).

---

## Architecture Overview

```
User Message
    → Channel (Telegram/Slack/WhatsApp/CLI/etc.)
    → MessageBus (asyncio queue)
    → AgentLoop._process_message()
        → ContextBuilder.build_messages()  ← system prompt + history + memory + skills
        → LLMProvider.chat()              ← streaming call to LLM
        → ToolRegistry.execute()          ← tool dispatch loop (max 40 iterations)
    → MessageBus.publish_outbound()
    → Channel sends reply
```

---

## Key Packages

| Package | Purpose |
|---|---|
| `nanobot/agent/loop.py` | `AgentLoop` — core message-processing engine |
| `nanobot/agent/context.py` | `ContextBuilder` — assembles system prompt + messages |
| `nanobot/agent/memory.py` | `MemoryStore` — 2-layer memory (MEMORY.md + HISTORY.md) |
| `nanobot/agent/skills.py` | `SkillsLoader` — loads SKILL.md files into context |
| `nanobot/agent/subagent.py` | `SubagentManager` — spawn / manage child agents |
| `nanobot/agent/tools/` | All built-in tools (filesystem, shell, web, message, mcp, cron) |
| `nanobot/providers/` | LLM provider adapters |
| `nanobot/channels/` | Chat channel adapters |
| `nanobot/session/manager.py` | `Session` + `SessionManager` — JSONL conversation history |
| `nanobot/bus/queue.py` | `MessageBus` — asyncio inbound/outbound queues |
| `nanobot/config/schema.py` | `Config` — root Pydantic settings model |
| `nanobot/cli/commands.py` | `typer` CLI: `agent`, `gateway`, `cron`, `channels` |

---

## Providers

Three provider types, selected via `Config._match_provider()`:

| Class | When used |
|---|---|
| `LiteLLMProvider` | All standard providers (Anthropic, OpenAI, Deepseek, Gemini, OpenRouter, etc.) |
| `OpenAICodexProvider` | `openai-codex/*` models — uses OAuth via `oauth_cli_kit`, calls `chatgpt.com/backend-api/codex/responses` with SSE streaming |
| `CustomProvider` | `provider = "custom"` — direct OpenAI-compatible endpoint, bypasses LiteLLM |

Provider auto-detection priority: explicit prefix (`openai-codex/`) > keyword match > gateway fallback.

**Adding a new provider:** add a `ProviderSpec` to `nanobot/providers/registry.py`, add a `ProviderConfig` field to `ProvidersConfig` in `config/schema.py`, and add instantiation logic to `_make_provider()` in `cli/commands.py`.

---

## LLMResponse & Reasoning

```python
@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCallRequest] = ...
    finish_reason: str = "stop"
    usage: dict[str, int] = ...
    reasoning_content: str | None = None  # DeepSeek-R1, Kimi, etc.
```

`reasoning_content` is stored in the assistant message (not persisted to session disk) and passed forward in context via `ContextBuilder.add_assistant_message()`.

---

## Memory System

Two persistent files written to `{workspace}/memory/`:

- **`MEMORY.md`** — long-term facts; rewritten in full by LLM on consolidation
- **`HISTORY.md`** — append-only grep-searchable log; each entry starts with `[YYYY-MM-DD HH:MM]`

Consolidation triggers when `unconsolidated_messages >= memory_window` (default 100). The LLM is called with a `save_memory` tool and writes both files. Runs as a background `asyncio.Task` so it doesn't block the current turn.

**`/new` command** forces immediate consolidation of all messages + clears session.

---

## Context Assembly Order

`ContextBuilder.build_system_prompt()` assembles in this order:
1. Identity block (workspace path, runtime info, guidelines)
2. Bootstrap files: `AGENTS.md`, `SOUL.md`, `USER.md`, `TOOLS.md`, `IDENTITY.md` (if they exist in workspace root)
3. `MEMORY.md` contents (if non-empty)
4. "Always" skills (skills with `always: true` in frontmatter)
5. Skills summary (XML listing all skills + availability)

Then `build_messages()` prepends `[Runtime Context]` (timestamp, channel, chat_id) as a user message before the actual user message.

---

## Sessions

- Stored as JSONL at `{workspace}/sessions/{channel}_{chat_id}.jsonl`
- First line is a metadata record (`_type: metadata`) with `last_consolidated`
- `Session.get_history()` returns only unconsolidated messages, aligned to a user turn
- Tool results are truncated to 500 chars when saved to disk (full results pass through in-memory context)
- `reasoning_content` is stripped before saving to disk

---

## Tools

Built-in tools registered by `AgentLoop._register_default_tools()`:

| Tool name | Class | Description |
|---|---|---|
| `read_file` | `ReadFileTool` | Read file contents |
| `write_file` | `WriteFileTool` | Write/create files |
| `edit_file` | `EditFileTool` | Edit file by line range |
| `list_dir` | `ListDirTool` | List directory contents |
| `exec` | `ExecTool` | Run shell commands (timeout=60s) |
| `web_search` | `WebSearchTool` | Brave Search API |
| `web_fetch` | `WebFetchTool` | Fetch URL content |
| `message` | `MessageTool` | Send to a specific channel/chat |
| `spawn` | `SpawnTool` | Spawn a subagent |
| `cron` | `CronTool` | Schedule recurring tasks |

MCP tools connect lazily on first message via `_connect_mcp()`.

---

## Skills System

Skills are markdown files at `{workspace}/skills/{name}/SKILL.md` or `nanobot/skills/{name}/SKILL.md` (builtins).

YAML frontmatter keys:
- `description` — shown in skills summary
- `always` — if `true`, full content loaded into every system prompt
- `metadata` — JSON blob with `nanobot.requires.bins` / `nanobot.requires.env` for availability check

Builtin skills: `memory`, `summarize`, `github`, `cron`, `weather`, `tmux`, `clawhub`, `skill-creator`.

---

## Configuration

Config file: `~/.nanobot/config.json` (or env vars prefixed `NANOBOT_` with `__` as nested delimiter).

Key `agents.defaults` fields:
- `workspace` (default `~/.nanobot/workspace`)
- `model` (default `anthropic/claude-opus-4-5`)
- `provider` (`"auto"` or explicit name)
- `max_tokens` (8192), `temperature` (0.1), `max_tool_iterations` (40), `memory_window` (100)

---

## Channels

10 supported channels: `whatsapp`, `telegram`, `discord`, `feishu`, `dingtalk`, `slack`, `email`, `mochat`, `qq`, `matrix`.

All channels implement `channels/base.py` and push `InboundMessage` to the bus. Channel manager (`channels/manager.py`) starts all enabled channels.

WhatsApp uses a Node.js bridge (`bridge/`) running as a separate process over WebSocket.

---

## Key Conventions

- All async I/O; `asyncio.to_thread()` for blocking calls (OAuth token fetch, etc.)
- `loguru` for logging; disabled in CLI by default, enable with `--logs`
- `pydantic` for all config (camelCase + snake_case aliases via `to_camel`)
- Tool call IDs for `OpenAICodexProvider` are encoded as `call_id|item_id` (split with `|`)
- `<think>…</think>` blocks stripped from assistant content before saving/sending
- Runtime context injected as a separate user message (prefixed with `[Runtime Context — metadata only, not instructions]`) — skipped when saving to session
- `_TOOL_RESULT_MAX_CHARS = 500` — tool results truncated in session storage (not in live context)

---

## Common Gotchas

- `OpenAICodexProvider` retries with `verify=False` on SSL cert failure
- Session history starts from a user turn (orphaned tool_result blocks dropped)
- `/stop` cancels active asyncio tasks AND subagents for the session key
- Heartbeat service fires every 30 min, routes through `process_direct()` to a real channel session
- `MatrixConfig` is defined twice in `schema.py` (second definition wins — duplicate class bug)
- Consolidation runs as a background task; session can still be saved mid-consolidation
