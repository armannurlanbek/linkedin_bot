# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A FastAPI app that generates Hebrew LinkedIn posts in the voice of Roman Binyaminov, CEO of Orlanda (an Israeli building-facade company). The core is an agentic loop: Claude Sonnet 4.6 with extended thinking calls tools (web scrape, web search, image search, RAG over the CEO's past posts, LinkedIn company lookup) and streams a finished post back to a single-page vanilla-JS frontend.

The system prompt that defines the writing style is a large **Hebrew** block in `app/services/agent.py::_build_system()`. Changing tone/structure/rules means editing that string — not config.

## Commands

Everything runs in Docker. The service is `api`, bound to `127.0.0.1:8082` (public access is via Cloudflare Tunnel, not direct exposure).

```bash
docker compose up -d --build          # build + run
docker compose logs -f api            # tail logs
docker compose restart api            # restart after code change (code is COPYed into the image, so rebuild for changes)
```

RAG / style data (run inside the container):

```bash
docker compose exec api python -m scripts.ingest          # embed posts.json into the `posts` table (idempotent: skips existing linkedin_id)
docker compose exec api python -m scripts.analyze_style   # regenerate style_card.json from all DB posts
docker compose exec api python -m scripts.pick_examples   # regenerate example_posts.json (golden style examples)
```

Local dev without Docker: uncomment `DATABASE_URL_SYNC` / `DATABASE_URL` in `.env` (see `.env.example`), then `uvicorn app.main:app --reload`.

There is **no test suite** and no linter configured. For a fast sanity check after editing Python, use `python -m py_compile <files>`.

## Secrets & config

Secrets live in `.env` (gitignored; see `.env.example`). The database URLs are **assembled in `docker-compose.yml`** from `DB_PASSWORD` — `.env` only holds the password, not the full URL (when running in Docker). Required keys: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` (embeddings + Whisper). `TAVILY_API_KEY` is optional — every Tavily-backed tool degrades to an empty result if it's missing. `APP_PASSWORD` empty disables the auth gate entirely.

## Architecture

### The agent loop — `app/services/agent.py::run_agent`
A **sync generator** that yields SSE strings (`_sse()` → `data: {json}\n\n`). It streams from Anthropic with extended thinking (budget 10k) and runs up to 8 tool iterations. The assistant message is only persisted **after** `stream.get_final_message()` completes — partial turns are never saved, which is why corruption from half-written assistant turns isn't a concern.

The system prompt is a single big block built in `_build_system()` and marked with `cache_control: ephemeral` for prompt caching. It embeds `style_card.json` (extracted style features) and `example_posts.json` (golden examples), both loaded fresh from disk on every request.

### Stream/persistence decoupling — `app/api/chats.py::_agent_stream`
`run_agent` runs in a **daemon worker thread with its own `SessionLocal()`**, pushing SSE events into an unbounded `queue.Queue`; the HTTP `StreamingResponse` just forwards from the queue. Consequence: **the turn completes and persists even if the client disconnects mid-stream** (e.g. a phone backgrounds Safari). The request-scoped `db` from `get_db` is *not* used by the agent — the worker owns its session because the request session closes at request end and isn't thread-safe. Don't "fix" this by passing the request `db` into the worker.

### Message persistence & replay
`messages.content` is **JSONB holding the raw Anthropic content-block list** (text / thinking / tool_use / tool_result / image / document). On replay, `_load_history`:
- runs `_sanitize_block` to strip SDK-internal fields the API rejects,
- drops empty text blocks (they cause 400s),
- and `_repair_tool_pairs` injects synthetic `(interrupted)` tool_results for any dangling `tool_use` so a disconnected turn doesn't 400 the next request.

Messages are ordered by **`(created_at, id)`** in both `_load_history` and `get_chat`. The `id` tiebreaker is load-bearing: equal timestamps under rapid tool loops could otherwise invert a `tool_use`/`tool_result` pair and 400 the API. Keep both order clauses in sync.

### Tools — `app/services/tools.py` (schemas + `execute_tool` dispatcher)
- `scrape_url` → `scraper.py`: httpx first, then Tavily *extract*, then Tavily *search* index as fallback; social/login-walled domains go straight to Tavily.
- `search_web`, `find_linkedin_profiles`, `search_images` → `search.py` (all Tavily).
- `retrieve_similar_posts` → `retrieval.py`: OpenAI `text-embedding-3-small` + pgvector cosine distance over the `posts` table.

### LinkedIn @mention flow (non-obvious)
`find_linkedin_profiles` is meant to be called **once**, for the single most important company. The branching lives in `run_agent` (not `execute_tool`):
- `count == 1` → emit `linkedin_resolved` SSE; Claude writes `@Company`.
- `count > 1` → emit `linkedin_disambiguation` SSE **and override the tool_result content** with a hard-stop instruction so Claude stops and does *not* write the post. The frontend shows a picker; the user's choice is sent back as a new message starting with `[LinkedIn @mentions confirmed]`, which the prompt tells Claude to act on immediately with no further tools.

### Data model — `app/db.py`
`posts` (RAG archive, `Vector(1536)`, unique `linkedin_id`), `chats`, `messages` (JSONB), `library_posts` (saved drafts; `promote` embeds the text and inserts it into `posts` to feed future RAG). **No Alembic** — `init_db()` does a best-effort `CREATE EXTENSION vector`, `create_all`, then `_migrate()`, a list of idempotent `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statements run on every startup. Add new columns there.

The app connects to a **host Postgres cluster through `db-router` (HAProxy)** as user `roman` → database `db_roman`, over the external Docker network `orlanda`. The `roman` role is not a superuser (hence the best-effort extension creation).

### Auth — `app/api/auth.py` + middleware in `app/main.py`
Shared-password gate backed by a signed HMAC cookie with a sliding 3-day window. Middleware protects `/api/*` except `/api/auth/*`; `/`, `/static/*`, `/health`, `/healthz` are always open. `/healthz` is a liveness probe that does **not** touch the DB; `/health` does.

### Frontend — `app/static/index.html`
A single ~2300-line vanilla-JS file, no build step. Multi-panel model: `chatPanels` keeps one DOM panel per chat in memory and toggles visibility; `chatStreams` maps chatId → `AbortController`; `activeChatId` tracks the visible one. It consumes the stream via **`fetch` + a `ReadableStream` reader** (not `EventSource`), splitting on `\n\n` and dispatching frames through `handleSSEEvent`.

Interruption handling mirrors the backend's finish-on-disconnect: a `TypeError` / "Load failed" / "Failed to fetch" is treated as a benign interruption (`markBackground` shows a calm amber note, chat added to `interruptedChats`) rather than a red error. `recoverInterruptedChat` then polls `GET /api/chats/{id}` — triggered on `visibilitychange→visible` and when re-opening the chat — and rebuilds the panel once the finished assistant reply has landed.
