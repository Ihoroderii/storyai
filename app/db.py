import json
import sqlite3
from pathlib import Path
from typing import Iterable, List, Tuple


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

CREATE TABLE IF NOT EXISTS visual_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter INTEGER NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL
);
"""

_MIGRATE_VISUAL_AUDIT = """
CREATE TABLE IF NOT EXISTS visual_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter INTEGER NOT NULL,
  kind TEXT NOT NULL,
  detail TEXT NOT NULL
);
"""


class NovelDB:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self):
        """Add new tables/columns to existing databases."""
        self.conn.executescript(_MIGRATE_VISUAL_AUDIT)

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

    def store_visual_audit(
        self,
        chapter: int,
        visual_notes: List[str],
        continuity_conflicts: List[str],
    ) -> None:
        """Persist visual notes and continuity conflicts for a chapter.

        Replaces any existing entries for this chapter.
        """
        self.conn.execute("DELETE FROM visual_audit WHERE chapter = ?", (chapter,))
        rows = []
        for note in visual_notes:
            rows.append((chapter, "visual_note", note))
        for conflict in continuity_conflicts:
            rows.append((chapter, "continuity_conflict", conflict))
        if rows:
            self.conn.executemany(
                "INSERT INTO visual_audit (chapter, kind, detail) VALUES (?, ?, ?)",
                rows,
            )
        self.conn.commit()

    def get_visual_audit(self, chapter: int | None = None) -> list[dict]:
        """Retrieve visual audit entries, optionally filtered by chapter."""
        if chapter is not None:
            rows = self.conn.execute(
                "SELECT chapter, kind, detail FROM visual_audit WHERE chapter = ? ORDER BY id",
                (chapter,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT chapter, kind, detail FROM visual_audit ORDER BY chapter, id"
            ).fetchall()
        return [{"chapter": r[0], "kind": r[1], "detail": r[2]} for r in rows]

    def get_continuity_conflicts(self, limit: int = 50) -> list[tuple[int, str]]:
        """Get recent continuity conflicts across all chapters."""
        rows = self.conn.execute(
            """
            SELECT chapter, detail FROM visual_audit
            WHERE kind = 'continuity_conflict'
            ORDER BY id DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows[::-1]

    def add_facts(self, facts: Iterable[Tuple[str, int]]) -> None:
        facts_list = list(facts)
        if not facts_list:
            return
        source_chapter = facts_list[0][1]
        self.conn.execute(
            "DELETE FROM canon_facts WHERE source_chapter = ?",
            (source_chapter,),
        )
        self.conn.executemany(
            "INSERT INTO canon_facts (fact, source_chapter) VALUES (?, ?)",
            facts_list,
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
