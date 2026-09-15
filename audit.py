"""
DataMind — audit logging.

Every question asked, every attempt the agent made (including failed
ones), and the final outcome gets written here permanently. This file
only ever INSERTS new rows — never updates or deletes existing ones —
which is what "append-only" means and why this is trustworthy as an
audit trail rather than just a log that could be quietly edited later.

Kept in its own file, separate from main.py, for the same reason as
sandbox.py: it has no dependency on FastAPI or Gemini, so it's easy to
test and reason about on its own.
"""

import json
import sqlite3
from datetime import datetime, timezone

DB_PATH = "audit_log.db"


def init_db():
    """
    Creates the audit_log table if it doesn't already exist yet. Safe
    to call every time the server starts up — "CREATE TABLE IF NOT
    EXISTS" simply does nothing if the table is already there, so this
    won't wipe out previous logs on restart.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            status TEXT NOT NULL,
            final_code TEXT,
            result TEXT,
            attempts TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_interaction(question: str, status: str, final_code: str, result, attempts: list):
    """
    Writes one permanent row recording exactly what happened for a
    single question — what was asked, the final outcome, and every
    attempt along the way. `result` and `attempts` are stored as JSON
    text, since SQLite doesn't have a native list/dict column type.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        INSERT INTO audit_log (timestamp, question, status, final_code, result, attempts)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            status,
            final_code,
            json.dumps(result),
            json.dumps(attempts),
        ),
    )
    conn.commit()
    conn.close()


def get_audit_log(limit: int = 50):
    """
    Returns the most recent audit log entries, newest first. This is
    what powers the /audit-log endpoint — anyone (you, a professor, an
    interviewer) can see the agent's full history, not just its most
    recent answer.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # lets us access columns by name below
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    entries = []
    for row in rows:
        entries.append({
            "id": row["id"],
            "timestamp": row["timestamp"],
            "question": row["question"],
            "status": row["status"],
            "final_code": row["final_code"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "attempts": json.loads(row["attempts"]),
        })
    return entries