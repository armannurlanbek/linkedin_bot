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

Importing any `app.*` module instantiates `app.config.Settings()` at import time, which **requires** `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `DATABASE_URL`, `DATABASE_URL_SYNC` to be present (env or `.env`) or it raises immediately. To unit-test a pure helper outside Docker (e.g. a scraper/search filter) without real secrets, set dummy values first — the helpers don't touch the DB or APIs:

```bash
ANTHROPIC_API_KEY=x OPENAI_API_KEY=x DATABASE_URL=postgresql://x DATABASE_URL_SYNC=postgresql://x python your_test.py
```

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
- `scrape_url` → `scraper.py`: httpx first, then Tavily *extract*, then Tavily *search* index as fallback; social/login-walled domains go straight to Tavily. Extracted images come from `og:image`/`<img>` tags filtered by `_SKIP_PATTERNS`. Trap: LinkedIn is *not* in the login-wall list, so a logged-out LinkedIn URL is scraped directly and its `og:image` resolves to the **publisher's company cover** (`media.licdn.com/.../company-background`), not the post's photo — which is why `_SKIP_PATTERNS` drops `licdn`/`linkedin` assets. Match that CDN host (`licdn.com`), not just `linkedin.com`, when filtering LinkedIn images anywhere.
- `search_web`, `find_linkedin_profiles`, `search_images` → `search.py` (all Tavily).

**Instagram/Facebook (`scraper.py`, social branch) — the user-agent trap.** Instagram renders `/p/<code>/embed/captioned/` server-side (caption + every carousel photo), but **only for a plain `Mozilla/5.0`**. The module's browser-like `HEADERS` string gets a 618 KB JavaScript shell with nothing in it — measured from production, 0/4 posts with `HEADERS` vs 4/4 with `_PLAIN_UA`, at ~0.5s against 8-15s for a Tavily extract. So Instagram goes to the embed endpoint first and only falls back to Tavily. Don't "fix" the embed fetch by giving it the realistic UA; that is the thing being refused.

Facebook has no equivalent — every user agent gets the login wall. But a `share/p/<code>` link 302s to its canonical `story.php` permalink in a 0-byte response, and Tavily reads the two shapes with **independent** success (on real URLs, each rescued a post the other returned nothing for), so `_resolve_facebook_share` follows redirects by hand and both shapes are tried.

**Do not add cookie-based or headless-browser scraping of these sites.** It requires a real logged-in account, which Meta will eventually checkpoint or disable — and that account would be the CEO's or the company's. If the logged-out path is not enough, the next step is a logged-out third-party scraper API (e.g. Apify), not credentials.
- `retrieve_similar_posts` → `retrieval.py`: OpenAI `text-embedding-3-small` + pgvector cosine distance over the `posts` table.

### Chat metadata — title, cover image, posted (`app/services/chat_meta.py`)
The sidebar title and cover image are derived from the **finished post**, not from the opening message. The opening message is usually a bare URL, which is why titles used to read "קישור לינקדאין קצר" (or a truncated refusal) and covers showed whatever photo the scraped page happened to carry.

- `run_agent` sets a cheap placeholder up front (`_PLACEHOLDER_TITLE` for a URL-only opening, otherwise `_generate_title`), then calls `finalize_chat_meta` **after** yielding `done` on the `end_turn` branch. Running it there keeps the post card finalising instantly, and it still completes on a disconnect because it is inside the worker thread.
- Cover selection is a priority rule, not a guess: `search_images` results beat `scrape_url` results, because `search_images` is queried with the building's name and ranked in `search.py::_rank_images`, while `scrape_url` images come from the source page. `run_agent` accumulates `image_candidates` per tool during the turn; when the post comes from a turn that ran no image tools (the second half of the disambiguation flow), `candidates_from_db` recovers them from the `__images__` kept in stored tool_results.
- **`title_source` / `thumbnail_source` gate every automatic write.** `manual` (the user renamed the chat or picked a cover via `PATCH /api/chats/{id}`) is never overwritten; `post`/`auto` means already settled; `auto`/`heuristic`/NULL means still eligible. The mid-run thumbnail write is additionally `WHERE thumbnail_url IS NULL`, so a later turn that scrapes something cannot clobber a good cover.
- **All metadata writes must go through `apply_chat_meta`**, which uses raw SQL. `chats.updated_at` carries `onupdate=func.now()` and the sidebar is ordered by it, so an ORM write here — above all the backfill's — would reshuffle the entire chat list into the order rows were touched.
- `scripts/backfill_chat_meta.py` applies the same functions to existing chats (`--dry-run` first; per-chat commit, so it is resumable and re-runnable).

### Posted vs Library
Publication state lives on **`chats`** (`posted_at`, `posted_message_id`), not on `library_posts` — the user's unit is the conversation, and most chats have no library row. `PATCH /api/chats/{id} {"posted": true}` resolves the chat's latest *finished* post (`is_post_message`: an assistant message with no `tool_use` and ≥300 chars of text, which is what excludes the short disambiguation hard-stop) and then calls `library.promote_item` so a published post also joins the RAG archive. The sidebar has three tabs: Chats, Posted, Library. `library_posts.status` and its PATCH endpoint still exist and still work, but the UI no longer writes them — the Library is archive-only.

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

## Skill routing

When the user's request matches an available skill, invoke it via the Skill tool. When in doubt, invoke the skill.

Key routing rules:
- Product ideas/brainstorming → invoke /office-hours
- Strategy/scope → invoke /plan-ceo-review
- Architecture → invoke /plan-eng-review
- Design system/plan review → invoke /design-consultation or /plan-design-review
- Full review pipeline → invoke /autoplan
- Bugs/errors → invoke /investigate
- QA/testing site behavior → invoke /qa or /qa-only
- Code review/diff check → invoke /review
- Visual polish → invoke /design-review
- Ship/deploy/PR → invoke /ship or /land-and-deploy
- Save progress → invoke /context-save
- Resume context → invoke /context-restore
- Author a backlog-ready spec/issue → invoke /spec
