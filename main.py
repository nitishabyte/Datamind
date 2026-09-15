"""
DataMind — Step 1: Basic FastAPI server

This is the smallest possible working server. Its only job right now
is to prove that:
1. FastAPI is installed and working
2. Your server can receive a request and send back a response

Everything else (file upload, AI, charts) gets added on top of this
in later steps.
"""

import io
import json
import os
import uuid

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel

from audit import get_audit_log, init_db, log_interaction
from sandbox import CodeTimeoutError, execute_pandas_code, make_json_safe

# Loads GEMINI_API_KEY from your .env file into the environment,
# exactly like in test_llm.py.
load_dotenv()

# This creates your "app" — think of it as the whole server object.
# Every route (URL) you add below gets attached to this app.
app = FastAPI(title="DataMind")

# Folder where generated chart images get saved. We create it upfront
# (if it doesn't already exist) because StaticFiles below needs the
# folder to actually exist at startup, and because generated code will
# be saving PNG files into it.
os.makedirs("charts", exist_ok=True)

# This makes everything inside the "charts" folder accessible directly
# over the web at /charts/<filename> — so once a chart is saved there,
# a browser (or your future frontend) can just load it like any normal
# image URL, no extra route needed for serving the file itself.
app.mount("/charts", StaticFiles(directory="charts"), name="charts")

# Same idea, but for the frontend's CSS and JS files — makes them
# reachable at /static/style.css and /static/app.js, which is exactly
# what index.html links to.
app.mount("/static", StaticFiles(directory="static"), name="static")

# One shared client for talking to Gemini, created once when the server
# starts rather than every time a question comes in. Reusing it is more
# efficient and is the standard pattern for any API client.
_gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Creates the audit_log table (if it doesn't already exist) the moment
# the server starts, so it's always ready before any question comes in.
init_db()

# This is a very simple form of "storage" — just a Python dictionary
# living in the server's memory. As soon as you stop the server, this
# data disappears. That's fine for now — in a later step we could swap
# this for a real database without changing anything else, because
# every other part of the code will just call get_current_dataframe().
_STORAGE = {"dataframe": None, "filename": None}


def format_profile_for_prompt(profile: dict) -> str:
    """
    Turns the structured profile dictionary (from Step 3) into a plain
    text description the AI can read, like a short paragraph a human
    analyst might write describing the dataset before diving in.

    Why we bother formatting it nicely instead of just dumping the raw
    JSON: AI models generally follow clear, human-readable instructions
    more reliably than they parse dense nested JSON. A few extra lines
    of formatting here noticeably improves how correct the generated
    code turns out to be.
    """
    lines = [f"Dataset with {profile['num_rows']} rows and {profile['num_columns']} columns."]

    for col, info in profile["columns"].items():
        if info.get("null_percentage") == 100:
            # Remember the empty-column issue we noticed in Step 3?
            # This is exactly where we fix it — we explicitly tell the
            # AI this column is empty, instead of letting it think a
            # blank column is a real numeric field.
            lines.append(f"- '{col}': EMPTY column (100% missing values) — ignore this column")
        elif info["type_category"] == "numeric":
            lines.append(
                f"- '{col}': numeric, {info['null_percentage']}% missing, "
                f"range {info.get('min')} to {info.get('max')}, mean {info.get('mean')}"
            )
        else:
            common_values = ", ".join(f"'{v}'" for v in info.get("most_common", {}).keys())
            lines.append(
                f"- '{col}': text/category, {info['null_percentage']}% missing, "
                f"{info['unique_values']} unique values, common values include: {common_values}"
            )

    return "\n".join(lines)


def build_code_prompt(question: str, profile_text: str) -> str:
    """
    Builds the actual instruction we send to the AI. Being explicit and
    strict about the rules here matters a LOT — vague instructions lead
    to code that doesn't run, or that the agent can't safely execute.
    This is what's called "prompt engineering" — just being precise
    about what you want.
    """
    return f"""You are a data analysis assistant. You are given a pandas DataFrame called `df` and a question about it.

Dataset schema:
{profile_text}

Question: {question}

Write ONLY Python pandas code that answers this question. Follow these rules exactly:
- The DataFrame is already loaded as `df` — do not re-create, re-load, or make up sample data.
- Store your final answer in a variable named `result`.
- Do not import anything — pandas is already available as `pd`.
- Do not include any explanation, comments, or markdown formatting like ``` — output ONLY the raw Python code, nothing else.
- If a term in the question (like "efficient", "good", "best") is not a literal column name, define a clear, reasonable interpretation of it using the available columns (for example: "efficient" = below-average value in one column AND above-average in another), apply that definition in your code, and mention the definition you used as part of `result`.
- Only say the question cannot be answered if it truly requires information that isn't in the dataset at all — not just because a term isn't a literal column name.
- If the question asks for or clearly implies a chart, graph, plot, trend over time, distribution, or a comparison across categories, ALSO create a chart: matplotlib is available as `plt`, and a ready-made file path is available as `chart_path` — save your chart with `plt.savefig(chart_path)`. Still set `result` to a short text summary even when you create a chart. If no chart is needed, simply don't call plt.savefig at all.
"""


def build_fix_prompt(question: str, profile_text: str, previous_code: str, error_message: str) -> str:
    """
    This is the prompt used on a RETRY, after the first attempt's code
    failed to run. The key difference from build_code_prompt: we show
    the AI its own broken code AND the exact error it caused, so it can
    reason about what went wrong — this is what makes this genuinely
    agentic behavior rather than just "try again and hope."
    """
    return f"""You are a data analysis assistant. Your previous attempt to write pandas code for this question failed.

Dataset schema:
{profile_text}

Question: {question}

Your previous code:
{previous_code}

The error this code produced when run:
{error_message}

Fix the code so it correctly answers the question, avoiding the error above. Follow these rules exactly:
- The DataFrame is already loaded as `df` — do not re-create, re-load, or make up sample data.
- Store your final answer in a variable named `result`.
- Do not import anything — pandas is already available as `pd`.
- Do not include any explanation, comments, or markdown formatting like ``` — output ONLY the raw Python code, nothing else.
- If a chart is appropriate, matplotlib is available as `plt` and a file path is available as `chart_path` — save with `plt.savefig(chart_path)`.
"""


def clean_code_response(text: str) -> str:
    """
    AI models very often wrap code in markdown fences like:
        ```python
        ...code...
        ```
    even when explicitly told not to. Rather than fighting the model
    with more instructions, it's simpler and more reliable to just
    strip those fences off ourselves if they show up.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # drop the opening ``` (and optional "python")
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop the closing ```
        text = "\n".join(lines)
    return text.strip()


class QuestionRequest(BaseModel):
    """
    Defines the shape of data we expect in the request body for /ask.
    FastAPI uses this to automatically validate incoming requests —
    if someone sends a request without a "question" field, FastAPI
    rejects it with a clear error before our code even runs.
    """
    question: str


def profile_column(series: pd.Series) -> dict:
    """
    Looks at a single column and describes it: what type of data it holds,
    how many values are missing, and some basic stats depending on
    whether it's numbers or text/categories.

    Why we do this per-column instead of just dumping the whole table:
    the AI (in a later step) needs to know things like "is 'age' actually
    a number?" and "does 'city' have missing values?" before it can write
    correct code. This is exactly that information, pre-computed.
    """
    non_null = series.dropna()

    col_info = {
        "null_count": int(series.isna().sum()),
        "null_percentage": round(float(series.isna().mean() * 100), 2),
    }

    if pd.api.types.is_numeric_dtype(series):
        col_info["type_category"] = "numeric"
        if len(non_null) > 0:
            col_info["min"] = float(non_null.min())
            col_info["max"] = float(non_null.max())
            col_info["mean"] = round(float(non_null.mean()), 2)
    else:
        # Anything not a number (text, categories, etc.) gets treated
        # the same way here — we show how many DIFFERENT values exist,
        # and the 3 most common ones, so the AI understands what kind
        # of values actually live in this column (e.g. "city" has
        # "Bangalore", "Mumbai", "Delhi" — not free-form text).
        col_info["type_category"] = "text/categorical"
        col_info["unique_values"] = int(series.nunique())
        top_values = non_null.value_counts().head(3)
        col_info["most_common"] = {str(k): int(v) for k, v in top_values.items()}

    return col_info


def build_data_profile(df: pd.DataFrame) -> dict:
    """
    Builds the full profile for the whole dataset by profiling every
    column one at a time. This is the single function we'll reuse
    whenever the AI needs to understand "what does this data look like"
    before answering a question.
    """
    return {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "columns": {col: profile_column(df[col]) for col in df.columns},
    }


def get_current_dataframe() -> pd.DataFrame:
    """
    Small helper function so that later code (Step 3 onward) doesn't
    need to know HOW the data is stored — it just asks for it. If no
    file has been uploaded yet, we raise a clear error instead of
    letting the code crash with a confusing message later.
    """
    if _STORAGE["dataframe"] is None:
        raise HTTPException(
            status_code=400,
            detail="No file uploaded yet. Upload a CSV first via /upload.",
        )
    return _STORAGE["dataframe"]


# This decorator says: "when someone visits the homepage ('/'),
# run the function right below it."
@app.get("/")
def read_root():
    # Step 10 change: "/" used to return a plain JSON message just to
    # prove the server was alive. Now it serves the actual frontend
    # page instead — this is the normal way a web app's homepage works:
    # the browser asks for "/", and gets back real HTML to render,
    # rather than raw API data. The old JSON check now lives at
    # /health below, which is the more standard place for it anyway.
    return FileResponse("static/index.html")


# A simple health-check route. Real projects almost always have one of
# these — it's a quick way to check "is my server alive?" without
# doing anything complicated.
@app.get("/health")
def health_check():
    return {"status": "ok"}


# This is the important new route. "UploadFile = File(...)" tells FastAPI:
# "expect a file to be sent here, and give it to me as this parameter."
@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    # Basic sanity check — don't let someone upload a .png or .exe by mistake.
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    # file.read() gets the raw bytes of the uploaded file. We wrap it in
    # io.BytesIO so pandas can treat those raw bytes like a file it's
    # reading from disk, even though it never actually touched your disk.
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        # If the CSV is malformed, tell the user clearly instead of
        # crashing the server with a raw Python error.
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {e}")

    # Save it in our simple in-memory storage so other routes can use it.
    _STORAGE["dataframe"] = df
    _STORAGE["filename"] = file.filename

    # Send back a small summary so the user (or the frontend, later) gets
    # immediate confirmation of what was actually loaded.
    return {
        "filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
    }


# A quick route to double check what's currently loaded, without
# re-uploading. Useful for testing directly, and will be reused by the
# Data Profiler in Step 3.
@app.get("/current-data")
def current_data():
    df = get_current_dataframe()
    return {
        "filename": _STORAGE["filename"],
        "rows": len(df),
        "columns": list(df.columns),
        # We use pandas' own to_json() here instead of to_dict(). Pandas
        # stores numbers using its own types (like numpy.int64) which
        # look like normal numbers but FastAPI's default JSON converter
        # doesn't recognize them and crashes. pandas' to_json() already
        # knows how to convert its own types safely, so we let it do
        # that conversion, then use json.loads() to turn that JSON text
        # back into a normal Python object that FastAPI CAN send.
        "preview": json.loads(df.head(5).to_json(orient="records")),
    }


# This route exposes the Data Profiler (Step 3) directly, so you can
# inspect it on its own. It's also reused internally by /ask below.
@app.get("/profile")
def get_profile():
    df = get_current_dataframe()
    return build_data_profile(df)


# This route now does the full Step 5 + 6 + 7 flow: generate code, try
# to run it, and if it fails, show the AI its own error and let it try
# again — up to MAX_ATTEMPTS times. This retry behavior is what makes
# this genuinely an agent rather than a one-shot text-to-code tool.
MAX_ATTEMPTS = 3


@app.post("/ask")
def ask_question(request: QuestionRequest):
    df = get_current_dataframe()
    profile = build_data_profile(df)
    profile_text = format_profile_for_prompt(profile)

    # A unique filename for this specific question's chart, so that
    # two people asking questions at the same time never overwrite
    # each other's chart file. We decide this BEFORE calling the AI,
    # so we can just hand it the path as a ready-made variable.
    chart_filename = f"{uuid.uuid4().hex}.png"
    chart_path = os.path.join("charts", chart_filename)

    # We keep a full record of every attempt — this isn't just for
    # debugging right now, it's exactly the structure Step 8 (the audit
    # logger) will save permanently, so a person can see the agent's
    # full reasoning trail, not just its final answer.
    attempts = []
    prompt = build_code_prompt(request.question, profile_text)

    for attempt_num in range(1, MAX_ATTEMPTS + 1):
        # This call is wrapped separately from the code-execution try/
        # except below, on purpose: a failure HERE means Gemini's own
        # servers had a problem (rate limits, temporary outages) — a
        # completely different situation from the AI successfully
        # replying but writing code that errors out. Treating them the
        # same would give a misleading error message.
        try:
            response = _gemini_client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )
        except Exception as e:
            log_interaction(
                question=request.question,
                status="ai_unavailable",
                final_code=None,
                result=None,
                attempts=attempts,
            )
            return {
                "question": request.question,
                "status": "ai_unavailable",
                "message": "The AI service is temporarily unavailable. Please try again in a moment.",
                "detail": str(e),
            }

        generated_code = clean_code_response(response.text)

        try:
            result = execute_pandas_code(generated_code, df, chart_path=chart_path)
            safe_result = make_json_safe(result)
            attempts.append({
                "attempt": attempt_num,
                "code": generated_code,
                "status": "success",
            })

            # Only include a chart_url in the response if the AI's code
            # actually called plt.savefig(chart_path) — otherwise this
            # file was never created, and we shouldn't claim one exists.
            chart_url = f"/charts/{chart_filename}" if os.path.exists(chart_path) else None

            # This is the actual audit step: permanently record what
            # was asked, what the agent finally did, and every attempt
            # it took to get there — including any earlier failures.
            log_interaction(
                question=request.question,
                status="success",
                final_code=generated_code,
                result=safe_result,
                attempts=attempts,
            )
            return {
                "question": request.question,
                "status": "success",
                "result": safe_result,
                "final_code": generated_code,
                "attempts": attempts,
                "chart_url": chart_url,
            }
        except Exception as e:
            error_message = str(e)
            attempts.append({
                "attempt": attempt_num,
                "code": generated_code,
                "status": "error",
                "error": error_message,
            })
            # Build the NEXT prompt using the fix-prompt, which includes
            # this failed code and its exact error — this is the actual
            # self-correction step. On the next loop iteration, the AI
            # sees exactly what went wrong.
            prompt = build_fix_prompt(request.question, profile_text, generated_code, error_message)

    # We only reach here if every attempt failed. Rather than a raw
    # crash, we return a clear, honest message plus the full attempt
    # history — a real agent should be able to say "I tried and
    # couldn't do this" instead of pretending to succeed.
    #
    # Logging failures too (not just successes) matters: a trustworthy
    # audit trail has to be honest about what DIDN'T work, not just a
    # highlight reel of correct answers.
    log_interaction(
        question=request.question,
        status="failed",
        final_code=None,
        result=None,
        attempts=attempts,
    )
    return {
        "question": request.question,
        "status": "failed",
        "message": f"Could not answer this question after {MAX_ATTEMPTS} attempts.",
        "attempts": attempts,
    }


# This route exposes the audit trail itself — anyone can see the full
# history of every question asked, what the agent ultimately did, and
# every attempt (successful or not) along the way. This is the feature
# that makes DataMind an inspectable, trustworthy agent rather than a
# black box.
@app.get("/audit-log")
def audit_log(limit: int = 50):
    return get_audit_log(limit=limit)