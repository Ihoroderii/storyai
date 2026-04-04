import sqlite3
from pathlib import Path
from typing import Iterable, Tuple


SCHEMA = """
CREATE TABLE IF NOT EXISTS canon_facts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fact TEXT NOT NULL,
  source_chapter INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chapter_summaries (
  chapter INTEGER PRIMARY KEY,
  summary_short TEXT NOT NULL,
  summary_detailed TEXT NOT NULL
);
"""


class NovelDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def upsert_summary(self, chapter: int, short: str, detailed: str) -> None:
        self.conn.execute(
            """
            INSERT INTO chapter_summaries (chapter, summary_short, summary_detailed)
            VALUES (?, ?, ?)
            ON CONFLICT(chapter) DO UPDATE SET
              summary_short=excluded.summary_short,
              summary_detailed=excluded.summary_detailed
            """,
            (chapter, short, detailed),
        )
        self.conn.commit()

    def add_facts(self, facts: Iterable[Tuple[str, int]]) -> None:
        self.conn.executemany(
            "INSERT INTO canon_facts (fact, source_chapter) VALUES (?, ?)",
            list(facts),
        )
        self.conn.commit()

    def get_recent_summaries(self, limit: int = 3) -> list[tuple[int, str]]:
        rows = self.conn.execute(
            """
            SELECT chapter, summary_short
            FROM chapter_summaries
            ORDER BY chapter DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows[::-1]

    def get_canon_facts(self, limit: int = 50) -> list[str]:
        rows = self.conn.execute(
            """
            SELECT fact FROM canon_facts
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [r[0] for r in rows][::-1]
