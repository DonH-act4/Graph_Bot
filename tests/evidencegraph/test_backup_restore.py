"""Restore checks use a disposable copy and reject broken backup references."""

import hashlib
import json
import runpy
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

script = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/verify_backup_restore.py")
)
verify_backup = cast(Callable[..., dict[str, int | bool]], script["verify_backup"])


def test_restored_backup_checks_databases_pdf_and_tutorial(tmp_path) -> None:
    backup = tmp_path / "backup"
    data = backup / "evidencegraph"
    data.mkdir(parents=True)
    pdf = b"%PDF-1.4\nsmall backup fixture"
    document_id = hashlib.sha256(pdf).hexdigest()
    paper = data / document_id
    paper.mkdir()
    (paper / "source.pdf").write_bytes(pdf)
    (data / "showcase.json").write_text(
        json.dumps({"thread_id": "sample", "document_id": document_id}), encoding="utf-8"
    )
    with sqlite3.connect(backup / "stage0-smoke.db") as connection:
        connection.execute("CREATE TABLE checkpoints (thread_id TEXT)")
    with sqlite3.connect(data / "conversations.sqlite3") as connection:
        connection.execute("CREATE TABLE conversations (thread_id TEXT, state TEXT)")
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?)",
            ("sample", json.dumps({"document_id": document_id})),
        )
    result = verify_backup(backup, require_pdfs=True)
    assert result == {
        "sqlite_files_checked": 2,
        "pdf_hashes_checked": 1,
        "tutorial_reference_valid": True,
        "full_paper_backup": True,
    }
    assert (paper / "source.pdf").read_bytes() == pdf


def test_restore_rejects_missing_tutorial_pdf(tmp_path) -> None:
    backup = tmp_path / "empty-backup"
    backup.mkdir()
    with pytest.raises(ValueError, match="missing a checkpoint"):
        verify_backup(backup, require_pdfs=True)


def test_restore_rejects_corrupted_pdf(tmp_path) -> None:
    backup = tmp_path / "backup"
    data = backup / "evidencegraph"
    data.mkdir(parents=True)
    pdf = b"%PDF-1.4\noriginal"
    document_id = hashlib.sha256(pdf).hexdigest()
    paper = data / document_id
    paper.mkdir()
    (paper / "source.pdf").write_bytes(b"%PDF-1.4\nchanged")
    (data / "showcase.json").write_text(
        json.dumps({"thread_id": "sample", "document_id": document_id}), encoding="utf-8"
    )
    with sqlite3.connect(backup / "stage0-smoke.db") as connection:
        connection.execute("CREATE TABLE checkpoints (thread_id TEXT)")
    with sqlite3.connect(data / "conversations.sqlite3") as connection:
        connection.execute("CREATE TABLE conversations (thread_id TEXT, state TEXT)")
        connection.execute(
            "INSERT INTO conversations VALUES (?, ?)",
            ("sample", json.dumps({"document_id": document_id})),
        )
    with pytest.raises(ValueError, match="PDF does not match"):
        verify_backup(backup, require_pdfs=True)


def test_restore_rejects_corrupted_database(tmp_path) -> None:
    backup = tmp_path / "backup"
    data = backup / "evidencegraph"
    data.mkdir(parents=True)
    (backup / "stage0-smoke.db").write_bytes(b"not a SQLite database")
    (data / "conversations.sqlite3").write_bytes(b"not a SQLite database")
    (data / "showcase.json").write_text("{}", encoding="utf-8")
    with pytest.raises(sqlite3.DatabaseError):
        verify_backup(backup)
