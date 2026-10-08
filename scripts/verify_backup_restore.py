"""Exercise an EvidenceGraph backup in a disposable directory, without touching live data."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path
from urllib.parse import quote


def _read_only_database(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{quote(str(path))}?mode=ro&immutable=1", uri=True)


def verify_backup(source: Path, *, require_pdfs: bool = False) -> dict[str, int | bool]:
    """Copy a backup, check SQLite integrity and saved Tutorial references, then discard copy."""
    if not source.is_dir():
        raise ValueError("Backup directory does not exist")
    required = (
        Path("stage0-smoke.db"),
        Path("evidencegraph/conversations.sqlite3"),
        Path("evidencegraph/showcase.json"),
    )
    if any(not (source / path).is_file() for path in required):
        raise ValueError("Backup is missing a checkpoint, conversation database, or Tutorial index")

    with tempfile.TemporaryDirectory(prefix="evidencegraph-restore-") as temporary:
        restored = Path(temporary) / "restored"
        shutil.copytree(source, restored)
        databases = sorted(
            path for path in restored.rglob("*") if path.suffix in {".db", ".sqlite3"}
        )
        for path in databases:
            with closing(_read_only_database(path)) as connection:
                result = connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise ValueError("Restored SQLite database failed integrity_check")

        showcase = json.loads((restored / required[2]).read_text(encoding="utf-8"))
        document_id = showcase.get("document_id")
        thread_id = showcase.get("thread_id")
        if not isinstance(document_id, str) or not isinstance(thread_id, str):
            raise ValueError("Tutorial index is incomplete")
        with closing(_read_only_database(restored / required[1])) as connection:
            row = connection.execute(
                "SELECT state FROM conversations WHERE thread_id = ?", (thread_id,)
            ).fetchone()
        if row is None or json.loads(row[0]).get("document_id") != document_id:
            raise ValueError("Tutorial index does not match the restored conversation")

        pdfs = sorted((restored / "evidencegraph").glob("*/source.pdf"))
        for path in pdfs:
            if hashlib.sha256(path.read_bytes()).hexdigest() != path.parent.name:
                raise ValueError("Restored PDF does not match its document identifier")
        if require_pdfs and not (restored / "evidencegraph" / document_id / "source.pdf").is_file():
            raise ValueError("Full backup is missing the Tutorial PDF")
        return {
            "sqlite_files_checked": len(databases),
            "pdf_hashes_checked": len(pdfs),
            "tutorial_reference_valid": True,
            "full_paper_backup": bool(pdfs),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument("--require-pdfs", action="store_true")
    args = parser.parse_args()
    try:
        result = verify_backup(args.backup, require_pdfs=args.require_pdfs)
    except (OSError, ValueError, sqlite3.DatabaseError, json.JSONDecodeError) as error:
        parser.exit(1, f"Restore verification failed: {error}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
