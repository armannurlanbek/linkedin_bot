I need to build a LinkedIn post generation agent for a CEO of a construction company. 
He specializes in building envelopes (facades, curtain walls, cladding systems).

His posts are in Hebrew (RTL). The agent must copy his writing style precisely.

---

## STACK
- Backend: FastAPI + Python
- LLM: claude-sonnet-4-6 (Anthropic)
- Embeddings: text-embedding-3-small (OpenAI) 
- Vector DB: pgvector (PostgreSQL)
- Containerization: Docker Compose
- Frontend: to be built later in Lovable

---

## PROJECT STRUCTURE TO CREATE
Set up the full project scaffold including:
- FastAPI app with proper folder structure
- Docker Compose with FastAPI + PostgreSQL + pgvector
- .env file template
- requirements.txt

---

## PHASE 1 — CSV INGESTION + STYLE ANALYSIS (build this first)

I have a CSV of 300+ LinkedIn posts written by the CEO. 

Build a script `ingest.py` that:
1. Reads the CSV (inspect columns first and handle accordingly)
2. Cleans each post (remove duplicates, empty posts, very short posts under 50 chars)
3. For each post, generates an embedding using text-embedding-3-small
4. Stores in pgvector with columns: id, text, embedding, char_count, created_at

Then build a script `analyze_style.py` that reads all posts and calls claude-sonnet-4-6 
to produce a STYLE CARD — a structured analysis of:
- Typical post length (short/medium/long)
- How he opens posts (question? bold statement? stat? story?)
- Sentence structure (short punchy vs long flowing)
- Emoji usage (which ones, how often, where)
- Hashtag patterns (how many, where placed, which topics)
- Tone (authoritative? conversational? inspirational?)
- Technical vocabulary he uses for building envelopes/facades
- How he closes posts (CTA? question? statement?)
- Hebrew/English mixing patterns
- Overall readability level (simple vs complex words)

Save the style card as `style_card.json` — we will inject this into every generation prompt.

---

## PHASE 2 — GENERATION ENDPOINT (build after Phase 1 is confirmed working)

POST /generate endpoint that accepts a URL and:
1. Scrapes the article (use httpx + BeautifulSoup)
2. Searches the web for additional project info (developer, contractor, architect)
3. Retrieves 5 most similar past posts from pgvector using the article summary as query
4. Generates a LinkedIn post using claude-sonnet-4-6 with:
   - The style card injected as system prompt
   - 5 retrieved example posts shown as demonstrations
   - Instruction to write in Hebrew, RTL, simple conversational language
   - Instruction to focus on building envelope / facade technical details
   - Instruction to always end with credits: Developer / Main Contractor / Architect
   - Instruction to use web-found data and cite sources
5. Returns: generated post text + list of image URLs from the article

---

## CRITICAL REQUIREMENTS
- Hebrew RTL must be preserved — never mix directions carelessly
- Language must be simple and easy to read — not overly formal or complex
- Always research the project online to find: developer, main contractor, architect
- Writing style must feel like the CEO wrote it himself, not like AI
- The style card is the source of truth for tone, structure, and vocabulary

---

Start with Phase 1 only. Do not build Phase 2 yet.
Show me the project structure first, then ask me to paste sample CSV rows 
so you can inspect the schema before writing the ingestion script.