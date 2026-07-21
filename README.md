# 📄 AI PDF Chat — Local RAG over PDFs

Upload a PDF, ask questions, get answers grounded in the document — with
**source page citations**. Runs fully locally, **no API keys needed**.

## Architecture

```
PDF upload
   │
   ▼
1. Extraction    PyMuPDF (fitz)            → text per page, block-sorted for
                                              sensible reading order
2. Chunking      paragraph/word-boundary    → chunks tagged with page number,
                 aware splitting              never cut mid-word
3. Embedding     sentence-transformers      → all-MiniLM-L6-v2 vectors
                 (all-MiniLM-L6-v2)
4. Indexing      FAISS (in-memory, cosine)  → IndexFlatIP
5. Retrieval     top-k similarity search    → candidate chunks
6. Generation    Two interchangeable engines →
                   - Groq (default if key set): hosted Llama 3.1/3.3,
                     fast + fluent + free tier
                   - Local Flan-T5-base: automatic fallback, no key,
                     lower quality
7. Citation      page numbers of the chunks → shown as plain text under the
                 actually used                answer, no PDF images rendered
```

This is a **retriever + reader** RAG pipeline — the same family as most
production RAG systems.

## Answer engine: Groq (recommended) or local fallback

The app has two interchangeable "readers":

* **Groq** — sends the retrieved chunks to Groq's hosted LLM (Llama 3.1/3.3)
over their API. Groq has a genuinely free tier, and inference is very
fast. This gives noticeably better, more fluent, more reliable answers.
Get a free key at https://console.groq.com → API Keys.
* **Local (Flan-T5-base)** — automatic fallback if no Groq key is entered.
Runs fully offline, no signup, no cost — but it's a small model and will
occasionally misread context or answer imprecisely.

Enter your Groq key in the sidebar text box at runtime — it's never
hardcoded or committed to the repo. For deployed apps, use your hosting
platform's secrets manager instead of typing it in every time:

* **Streamlit Community Cloud**: App settings → Secrets → add
`GROQ\_API\_KEY = "..."`, then read it in code with `st.secrets\["GROQ\_API\_KEY"]`
as a default value for the text input.
* **Locally**: create `.streamlit/secrets.toml` with the same line (this
file should be in `.gitignore` — never commit API keys).

## Why generation instead of extractive QA

An earlier version of this used DistilBERT-SQuAD (extractive QA) to pull an
exact text span out of a chunk as the "answer." That sounds appealing
(zero hallucination risk, since the answer is always a literal substring),
but in practice it broke down badly on real documents like resumes:

* DistilBERT-SQuAD was trained on Wikipedia trivia questions. Its
confidence scores are **not calibrated** for other document types — it
would confidently return a wrong span at 30-40% "confidence" just as
often as a right one, so a confidence threshold couldn't reliably filter
out bad answers.
* It also can't synthesize anything — "explain the projects" has no single
correct span to extract, so it always guessed.

Swapping in a small **generative** model (Flan-T5-base) that actually reads
the retrieved context and writes a response — with an explicit instruction
to say "I couldn't find this in the document" when the context doesn't
contain the answer — gives much more usable results for both narrow
factual questions and broader ones, while staying 100% local and free.

**Honest limitation to know about:** Flan-T5-base is a small (\~250M
parameter) model. It's far more capable than extractive QA for this task,
but it's not as fluent or reliable as a large hosted LLM (GPT-4, Claude,
Gemini, etc.) — it can still occasionally misread context or answer
imprecisely, especially on dense/technical text. This is the real
trade-off of staying fully local and free; the natural upgrade path is
swapping this function for a call to a hosted LLM API once you're open to
using one (see "Possible upgrades" below).

## Run locally

```bash
pip install -r requirements.txt
```

**Set your Groq key so it's never shown in the UI** — create a file named
`.env` in the project folder (copy `.env.example` and fill it in):

```
GROQ\_API\_KEY=your\_actual\_key\_here
```

This file is already in `.gitignore`, so it's never committed. Then run:

```bash
python -m streamlit run app.py  or streamlit run app.py
```

The sidebar will just say "Using Groq" — no password box shown, since the
key was found automatically. If you don't set up `.env` at all, the app
falls back to showing a manual key-entry box (handy for quick testing), and
if that's also left blank, it uses the local Flan-T5 model instead.

First run downloads the embedding model (\~90MB) always. The local Flan-T5
fallback (\~250MB) only downloads the first time you actually use the local
engine (i.e. no Groq key found anywhere) — if you always use Groq, it's
never downloaded.

## Deploy for free (pick one)

### Option A — Streamlit Community Cloud (easiest)

1. Push this folder to a public GitHub repo (`.env` won't be included,
thanks to `.gitignore` — double check it's not there before pushing).
2. Go to https://share.streamlit.io → "New app" → point it at your repo,
branch `main`, file `app.py`.
3. Before/after deploying, go to your app's **Settings → Secrets** and add:

```
   GROQ\_API\_KEY = "your\_actual\_key\_here"
   ```

   The app reads this automatically (`st.secrets\["GROQ\_API\_KEY"]`) — same
effect as the local `.env` file, no password box shown to visitors.

4. Deploy. First boot is slow (model download); subsequent loads are fast.

### Option B — Hugging Face Spaces (more headroom for ML apps)

1. Create a new Space → SDK: **Streamlit**.
2. Push this folder's contents to the Space's repo (`git push`).
3. In the Space's **Settings → Repository secrets**, add `GROQ\_API\_KEY`.
4. It builds automatically from `requirements.txt` + `app.py`.

Both are free tiers suitable for a portfolio/demo project.

## What to say about this in an interview

* **Why a retriever + local generator instead of extractive QA?** Extractive
QA models have miscalibrated confidence outside their training
distribution and can't synthesize answers to open-ended questions. A
small local generative model, prompted to answer only from retrieved
context (and to say when it can't), gives grounded answers without
needing an API key — a genuine RAG pattern, just with a lightweight
reader.
* **How would you extend this to be "agentic"?** Add a router/agent step
that decides retrieve-and-answer vs. summarize-whole-doc vs. multi-hop
retrieval (ask sub-questions, retrieve again, combine). Swap in a larger
hosted LLM (Claude/OpenAI/Gemini) for noticeably better answer quality
once an API key is acceptable — that's the most natural "Day 2" upgrade.
* **Why FAISS over a hosted vector DB?** Zero infra/cost for a single-user
demo; call out that a real product would swap in Pinecone/Weaviate/pgvector
for persistence across sessions and multiple documents.
* **Known limitations** (good to proactively mention): scanned/image-only
PDFs need OCR (not included); single document per session (no
persistence); the local generator is small and can occasionally misread
context, unlike a large hosted LLM.

## Possible upgrades (mention as roadmap, or build if you have time)

* Swap `generate\_answer` in `rag\_utils.py` for a call to a hosted LLM API
(Claude/OpenAI/Gemini) once you have a key — same retrieved-context
prompt, much stronger answers.
* Add OCR (e.g. `pytesseract`) for scanned PDFs.
* Persist the FAISS index + chunks to disk/SQLite so re-uploads aren't
needed between sessions.
* Multi-document support (chat across a whole folder of PDFs).

