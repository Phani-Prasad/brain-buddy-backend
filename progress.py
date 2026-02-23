"""
Brain Buddy - Progress Tracking Module
Logs user activity and generates dashboard statistics.
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

from auth import DB_PATH, get_db


# ── DB init ───────────────────────────────────────────────────────────────────
def init_progress_db():
    """Create activity_log table if it doesn't exist."""
    conn = get_db()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity_log (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                activity_type TEXT    NOT NULL,
                subject       TEXT    DEFAULT 'General',
                score         INTEGER DEFAULT NULL,
                metadata      TEXT    DEFAULT '{}',
                created_at    TEXT    NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ── Logging ───────────────────────────────────────────────────────────────────
def log_activity(
    user_id: int,
    activity_type: str,
    subject: str = "General",
    score: Optional[int] = None,
    metadata: Optional[dict] = None,
):
    """
    Log a user activity event.
    activity_type: 'chat_message' | 'flashcard_session' | 'quiz_attempt' | 'voice_session' | 'explanation'
    score: 0-100 (for quiz/flashcard sessions)
    """
    conn = get_db()
    try:
        conn.execute(
            """
            INSERT INTO activity_log (user_id, activity_type, subject, score, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                activity_type,
                subject,
                score,
                json.dumps(metadata or {}),
                datetime.utcnow().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ── Stats ─────────────────────────────────────────────────────────────────────
def get_user_stats(user_id: int) -> Dict:
    """Return aggregated stats for the dashboard."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM activity_log WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return _empty_stats()

    activities = [dict(r) for r in rows]

    # ── Totals ────────────────────────────────────────────────────────────────
    total_messages     = sum(1 for a in activities if a["activity_type"] == "chat_message")
    flashcard_sessions = sum(1 for a in activities if a["activity_type"] == "flashcard_session")
    voice_sessions     = sum(1 for a in activities if a["activity_type"] == "voice_session")
    explanations       = sum(1 for a in activities if a["activity_type"] == "explanation")
    quiz_attempts      = [a for a in activities if a["activity_type"] == "quiz_attempt"]
    quiz_scores        = [a["score"] for a in quiz_attempts if a["score"] is not None]
    avg_quiz_score     = round(sum(quiz_scores) / len(quiz_scores)) if quiz_scores else None

    # ── Flashcard accuracy (from metadata) ───────────────────────────────────
    fc_got   = sum(json.loads(a["metadata"]).get("got", 0) for a in activities if a["activity_type"] == "flashcard_session")
    fc_total = sum(json.loads(a["metadata"]).get("total", 0) for a in activities if a["activity_type"] == "flashcard_session")
    fc_accuracy = round((fc_got / fc_total) * 100) if fc_total > 0 else None

    # ── Subject breakdown ─────────────────────────────────────────────────────
    subject_counts: Dict[str, int] = {}
    for a in activities:
        s = a["subject"] or "General"
        subject_counts[s] = subject_counts.get(s, 0) + 1
    top_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)[:6]

    # ── Activity streak ───────────────────────────────────────────────────────
    streak = _calculate_streak(activities)

    # ── Activity this week (last 7 days) ──────────────────────────────────────
    now = datetime.utcnow()
    week_data = []
    for i in range(6, -1, -1):
        day = now - timedelta(days=i)
        day_str = day.strftime("%Y-%m-%d")
        count = sum(1 for a in activities if a["created_at"][:10] == day_str)
        week_data.append({"day": day.strftime("%a"), "count": count})

    # ── Recent activity feed (last 10) ────────────────────────────────────────
    recent = []
    for a in activities[:10]:
        recent.append({
            "type": a["activity_type"],
            "subject": a["subject"],
            "score": a["score"],
            "timestamp": a["created_at"],
        })

    return {
        "totals": {
            "messages":          total_messages,
            "flashcard_sessions": flashcard_sessions,
            "voice_sessions":    voice_sessions,
            "explanations":      explanations,
            "quiz_attempts":     len(quiz_attempts),
            "avg_quiz_score":    avg_quiz_score,
            "fc_accuracy":       fc_accuracy,
            "streak":            streak,
        },
        "subject_breakdown": [{"subject": s, "count": c} for s, c in top_subjects],
        "week_activity": week_data,
        "recent_activity": recent,
    }


def _calculate_streak(activities: List[Dict]) -> int:
    """Count how many consecutive days the user had at least 1 activity."""
    if not activities:
        return 0
    unique_days = sorted(
        {a["created_at"][:10] for a in activities}, reverse=True
    )
    streak = 0
    today = datetime.utcnow().date()
    for i, day_str in enumerate(unique_days):
        day = datetime.strptime(day_str, "%Y-%m-%d").date()
        expected = today - timedelta(days=i)
        if day == expected:
            streak += 1
        else:
            break
    return streak


def _empty_stats() -> Dict:
    """Return zeroed stats for users with no activity."""
    now = datetime.utcnow()
    return {
        "totals": {
            "messages": 0, "flashcard_sessions": 0, "voice_sessions": 0,
            "explanations": 0, "quiz_attempts": 0, "avg_quiz_score": None,
            "fc_accuracy": None, "streak": 0,
        },
        "subject_breakdown": [],
        "week_activity": [
            {"day": (now - timedelta(days=i)).strftime("%a"), "count": 0}
            for i in range(6, -1, -1)
        ],
        "recent_activity": [],
    }
