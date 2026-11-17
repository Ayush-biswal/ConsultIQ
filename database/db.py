import sqlite3
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "consultiq.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS consultations (
            consultation_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            description TEXT DEFAULT '',
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Active'
        );

        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY AUTOINCREMENT,
            consultation_id INTEGER NOT NULL,
            section_name TEXT NOT NULL,
            FOREIGN KEY (consultation_id) REFERENCES consultations(consultation_id)
                ON DELETE CASCADE,
            UNIQUE(consultation_id, section_name)
        );

        CREATE TABLE IF NOT EXISTS feedback (
            feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
            section_id INTEGER NOT NULL,
            comment TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'Manual',
            created_at TEXT NOT NULL,
            FOREIGN KEY (section_id) REFERENCES sections(section_id)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            feedback_id INTEGER NOT NULL UNIQUE,
            sentiment TEXT NOT NULL,
            positive_probability REAL NOT NULL,
            neutral_probability REAL NOT NULL,
            negative_probability REAL NOT NULL,
            is_priority INTEGER NOT NULL DEFAULT 0,
            priority_reason TEXT DEFAULT '',
            FOREIGN KEY (feedback_id) REFERENCES feedback(feedback_id)
                ON DELETE CASCADE
        );
        """)


def create_consultation(title, description, section_names):
    title = title.strip()
    if not title:
        raise ValueError("Consultation title is required.")
    clean_sections = list(dict.fromkeys(s.strip() for s in section_names if s.strip()))
    if not clean_sections:
        raise ValueError("Add at least one section.")

    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO consultations(title, description, created_at) VALUES (?, ?, ?)",
            (title, description.strip(), datetime.now().isoformat(timespec="seconds"))
        )
        consultation_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO sections(consultation_id, section_name) VALUES (?, ?)",
            [(consultation_id, section) for section in clean_sections]
        )
    return consultation_id


def get_consultations():
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM consultations ORDER BY consultation_id DESC"
        ).fetchall()]


def get_sections(consultation_id):
    with get_connection() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM sections WHERE consultation_id=? ORDER BY section_name",
            (consultation_id,)
        ).fetchall()]


def add_feedback(section_id, comment, source="Manual"):
    comment = comment.strip()
    if not comment:
        raise ValueError("Feedback comment cannot be empty.")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO feedback(section_id, comment, source, created_at) VALUES (?, ?, ?, ?)",
            (section_id, comment, source, datetime.now().isoformat(timespec="seconds"))
        )
        return cur.lastrowid


def save_prediction(feedback_id, sentiment, probabilities, is_priority, reason):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO predictions(
                feedback_id, sentiment, positive_probability, neutral_probability,
                negative_probability, is_priority, priority_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(feedback_id) DO UPDATE SET
                sentiment=excluded.sentiment,
                positive_probability=excluded.positive_probability,
                neutral_probability=excluded.neutral_probability,
                negative_probability=excluded.negative_probability,
                is_priority=excluded.is_priority,
                priority_reason=excluded.priority_reason
        """, (
            feedback_id, sentiment,
            float(probabilities.get("Positive", 0)),
            float(probabilities.get("Neutral", 0)),
            float(probabilities.get("Negative", 0)),
            int(bool(is_priority)), reason
        ))


def get_feedback(consultation_id=None, section_id=None):
    sql = """
        SELECT f.feedback_id, f.comment, f.source, f.created_at,
               s.section_name, c.consultation_id, c.title AS consultation_title,
               p.sentiment, p.positive_probability, p.neutral_probability,
               p.negative_probability, p.is_priority, p.priority_reason
        FROM feedback f
        JOIN sections s ON f.section_id=s.section_id
        JOIN consultations c ON s.consultation_id=c.consultation_id
        LEFT JOIN predictions p ON f.feedback_id=p.feedback_id
        WHERE 1=1
    """
    params = []
    if consultation_id is not None:
        sql += " AND c.consultation_id=?"
        params.append(consultation_id)
    if section_id is not None:
        sql += " AND s.section_id=?"
        params.append(section_id)
    sql += " ORDER BY f.feedback_id DESC"

    with get_connection() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


def get_section_id(consultation_id, section_name):
    with get_connection() as conn:
        row = conn.execute(
            "SELECT section_id FROM sections WHERE consultation_id=? AND section_name=?",
            (consultation_id, section_name)
        ).fetchone()
    return row["section_id"] if row else None
