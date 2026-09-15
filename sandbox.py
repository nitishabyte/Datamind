"""
DataMind — sandboxed code execution.

This is kept in its own file, separate from main.py, for two reasons:
1. It has no dependency on FastAPI/Gemini — it's pure "run this code
   safely" logic, which makes it easy to test on its own.
2. It uses Python's multiprocessing, which works more predictably when
   it's not tangled up with a web server's threading.

Why a separate PROCESS instead of just a timer:
A previous version of this used signal.alarm() as a timeout, but that
only works in Python's main thread — and FastAPI runs normal route
functions in a background thread, so it broke immediately in real use.
Running the code in a separate process instead fixes that, and adds a
genuine safety benefit: a process can be forcibly killed if it hangs,
whereas a Python thread cannot be force-stopped at all.
"""

import json
import multiprocessing as mp

import matplotlib
# "Agg" is a non-interactive backend — it renders charts straight to an
# image file instead of trying to pop up a window, which is what we
# need on a server (there's no screen to show a window on). This MUST
# be set before importing pyplot, or matplotlib may try to pick a
# graphical backend that doesn't exist here and error out.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class CodeTimeoutError(Exception):
    """Raised when generated code takes too long to run."""
    pass


# Only these built-in Python functions are available to AI-generated
# code. Everything not listed here (open, __import__, exec, eval,
# input, etc.) is unavailable, which blocks file/network/system access
# entirely — this is the core safety boundary for what the AI can do.
_SAFE_BUILTINS = {
    "len": len, "range": range, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "list": list, "dict": dict, "set": set,
    "tuple": tuple, "str": str, "int": int, "float": float,
    "bool": bool, "round": round, "enumerate": enumerate,
    "zip": zip, "abs": abs, "print": print,
}

# "fork" is explicitly requested here (rather than relying on whatever
# the operating system defaults to) because it's the most reliable,
# widely-supported way to spawn a lightweight child process on Linux,
# which is what your Codespace runs on.
_ctx = mp.get_context("fork")


def _worker(code: str, df: pd.DataFrame, chart_path: str, queue: mp.Queue):
    """
    This function runs INSIDE the separate process, not in your main
    server. It executes the AI's code and puts the outcome (success or
    error) onto a queue so the parent process can read it back.

    `chart_path` is a file path we've already decided on (in main.py)
    before the AI even runs. We hand it to the generated code as a
    ready-made variable, so the AI's instructions can simply say
    "save your chart to chart_path" without needing to invent a
    filename itself — one less thing that could go wrong.
    """
    try:
        sandbox_locals = {"df": df.copy(), "pd": pd, "plt": plt, "chart_path": chart_path}
        sandbox_globals = {"__builtins__": _SAFE_BUILTINS}
        exec(code, sandbox_globals, sandbox_locals)

        # Whether or not the code actually drew a chart, always close
        # any open matplotlib figures afterward. This is just good
        # hygiene — leftover open figures can silently eat up memory
        # over many requests if we don't clean up.
        plt.close("all")

        if "result" not in sandbox_locals:
            queue.put(("error", "The generated code did not set a variable named 'result'."))
        else:
            queue.put(("success", sandbox_locals["result"]))
    except Exception as e:
        plt.close("all")
        queue.put(("error", str(e)))


def execute_pandas_code(code: str, df: pd.DataFrame, chart_path: str, timeout_seconds: int = 5):
    """
    Runs AI-generated code in a separate, restricted process and
    returns whatever it stored in `result`. If the code also saved a
    chart image to `chart_path`, that file will exist on disk once
    this function returns — the caller (main.py) checks for it.

    Safety measures, each guarding a different risk:
    1. Restricted builtins (in _worker) — blocks file/network/system access.
    2. A real timeout with process.terminate() — if the code hangs or
       loops forever, we forcibly kill the process after timeout_seconds,
       so one bad question can never freeze your server.
    3. df.copy() inside the worker — your actual uploaded data in the
       main server process is never touched, even if generated code
       tries to modify or delete it.
    """
    queue = _ctx.Queue()
    process = _ctx.Process(target=_worker, args=(code, df, chart_path, queue))
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # The code is still running after our time limit — kill it.
        process.terminate()
        process.join()
        raise CodeTimeoutError(f"Code took longer than {timeout_seconds} seconds and was stopped.")

    if queue.empty():
        # This can happen if the process crashed in a way that never
        # reached our try/except (e.g. a segfault) — rare, but we
        # handle it rather than letting the server hang waiting.
        raise RuntimeError("Code execution failed unexpectedly with no error message.")

    status, payload = queue.get()
    if status == "error":
        raise RuntimeError(payload)
    return payload


def make_json_safe(value):
    """
    Converts whatever the executed code produced into something that
    can be safely turned into JSON. This function is RECURSIVE — it
    checks not just the top-level value, but everything nested inside
    dictionaries, lists, and arrays too. This matters because pandas
    operations like `.groupby(...).sum().to_dict()` produce a plain
    Python dict, but the NUMBERS inside that dict are still numpy types
    (numpy.int64, numpy.float64) — checking only the outer dict misses
    exactly this case.
    """
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="records"))
    if isinstance(value, pd.Series):
        return json.loads(value.to_json())
    if isinstance(value, pd.Index):
        return [make_json_safe(v) for v in value.tolist()]
    if isinstance(value, np.ndarray):
        return [make_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value