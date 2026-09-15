# DataMind

Upload a dataset, ask it questions in plain English, and get back a real answer — computed by actual generated code, not a hallucinated guess — along with a full, inspectable record of how the agent got there.

## What this is

Most "chat with your data" tools either hardcode a fixed set of charts, or blindly trust whatever an LLM says without ever checking if it's true. DataMind takes a different approach: every question is answered by **generating real pandas code, running it safely, and showing its work** — including every failed attempt along the way, not just the final answer.

## Why it exists

Two constraints shaped this project:

1. **Trust.** An AI agent that just states an answer is a black box. DataMind logs every question, every piece of generated code, every execution error, and every retry — permanently, in an audit trail anyone can inspect. Ambiguous terms in a question ("efficient", "best") aren't quietly guessed at either — the agent states the definition it used as part of its answer.
2. **Safety.** Letting an LLM's generated code run unsupervised is genuinely risky. DataMind executes all generated code in a locked-down subprocess with a restricted set of built-in functions, a hard timeout that can forcibly kill a hung process, and zero access to the filesystem or network.

## Features

- **Natural language → real analysis.** Ask a question in plain English, get back a computed answer, not a paraphrased guess.
- **Self-correcting agent loop.** If generated code fails, the agent sees its own error message and tries again (up to 3 attempts) before giving up honestly.
- **Sandboxed execution.** Generated code runs in an isolated subprocess with a restricted built-in set, a real timeout, and no access to the real dataset (only a copy).
- **Full audit trail.** Every question, every attempt, every outcome — success or failure — is permanently logged and viewable via the UI or `/audit-log`.
- **Automatic chart generation.** When a question implies a visual answer, the agent generates and returns a real matplotlib chart.
- **Handles genuinely messy real data** — missing values, all-empty columns, mixed types — not just clean tutorial CSVs.

## Architecture

```
                 FRONTEND
          HTML + CSS + JavaScript
                    │
                    │ HTTP Requests
                    ↓
                 FASTAPI
                BACKEND API
                    │
        ┌───────────┼─────────────┬──────────────┐
        ↓           ↓             ↓              ↓
   Data Profiler  Prompt      Sandboxed      Chart
  (schema/stats)  Builder     Executor      Renderer
        │           │             │              │
        └─────→ Gemini API   (subprocess,    (matplotlib,
                (code gen)    timeout,         Agg backend)
                              restricted
                              builtins)
                    │
                    ↓
          Self-Correction Loop
           (on execution error)
                    │
                    ↓
              Audit Logger
             (SQLite, append-only)
```

**Why each piece exists:**

- **Data Profiler** inspects the actual uploaded dataset's schema, types, and null rates *before* any code gets generated — this is what stops the AI from guessing wrong column names or treating an empty column as numeric.
- **Sandboxed Executor** runs in a separate OS process, not just a thread, specifically so it can be forcibly killed if generated code hangs — a thread in Python cannot be force-stopped, a process can.
- **Self-Correction Loop** re-prompts the AI with its own previous code and the exact error it caused, rather than giving up after one failure — this is what makes the system agentic rather than a one-shot text-to-code tool.
- **Audit Logger** writes every interaction permanently and append-only, so the agent's reasoning is inspectable after the fact, not just visible in the moment.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python) |
| AI | Google Gemini API |
| Data handling | pandas |
| Sandboxing | Python `multiprocessing` |
| Charts | matplotlib (Agg backend) |
| Audit storage | SQLite |
| Frontend | Vanilla HTML/CSS/JS |

## How a question actually gets answered

1. The uploaded CSV is profiled — column types, null rates, sample values.
2. That profile plus the question is sent to Gemini, which returns pandas code.
3. The code runs in a sandboxed subprocess with a 5-second timeout and a restricted set of built-ins (no file access, no imports, no network).
4. If it fails, the exact error is shown back to the AI, which tries again (up to 3 times).
5. If it succeeds, the result — and a chart, if one was generated — is returned and permanently logged.

## Setup

```bash
git clone <your-repo-url>
cd datamind
python -m venv venv
source venv/bin/activate   # venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file (see `.env.example` for the format) with your Gemini API key:
```
GEMINI_API_KEY=your-key-here
```

Run the server:
```bash
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

## Project structure

```
datamind/
├── main.py          # FastAPI app, routes, prompt construction
├── sandbox.py        # Safe code execution (subprocess + timeout + restricted builtins)
├── audit.py          # Append-only audit logging (SQLite)
├── static/           # Frontend (HTML/CSS/JS)
├── requirements.txt
├── .env.example
└── .gitignore
```

## Engineering challenges solved along the way

A few real, non-obvious bugs came up while building this — worth noting since they're more interesting than the happy path:

- **`signal.alarm` doesn't work from a background thread.** FastAPI runs synchronous route handlers in a thread pool, not the main thread, so the original timeout approach silently broke. Fixed by moving execution into a genuinely separate OS process instead, which can also be forcibly terminated — something a thread cannot.
- **Numpy types nested inside dicts/lists broke JSON serialization.** `pandas` operations like `.unique()` or `.value_counts().to_dict()` can return `numpy.int64`/`numpy.ndarray` values buried inside otherwise-normal Python containers. Fixed with a recursive JSON-safety converter that checks every nested value, not just the top-level type.
- **The agent initially refused ambiguous questions instead of reasoning about them.** Asking "which classrooms are efficient?" against a dataset with no "efficiency" column caused the agent to give up, technically correctly but unhelpfully. Fixed by explicitly instructing the agent to define and state a reasonable interpretation of ambiguous terms rather than refusing outright.

## Future improvements

- Session-level memory, so a defined term (e.g. "efficient") stays consistent across a conversation instead of being redefined per question
- Support for connecting directly to a database instead of only CSV uploads
- Streaming responses instead of waiting for the full agent loop to finish