"""Immutable receipts and a durable rollout ledger. Read-only audits never open SQLite."""
import contextlib
import os
import sqlite3
import time
from pathlib import Path
from .identity import canonical, digest, read, seal, verify, inside, file_sha


def immutable(path, value):
    path = inside(path)
    if path.exists():
        raise RuntimeError("Existing artifact blocks overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(canonical(value) + "\n")
        f.flush()
        os.fsync(f.fileno())


@contextlib.contextmanager
def lock(directory):
    directory = inside(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "writer.lock").open("a+b") as f:
        if f.seek(0, 2) == 0:
            f.write(b"0"); f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@contextlib.contextmanager
def stage(run, name, lineage):
    directory = inside(Path(run) / "stages" / name)
    if directory.exists():
        raise RuntimeError("Stage already exists; no overwrite or automatic retry")
    immutable(directory / "started.json", seal({"stage": name, "lineage": lineage,
                                               "started_utc_epoch": time.time()}))
    try:
        yield directory
    except BaseException as exc:
        immutable(directory / "failed.json", seal({"error_type": type(exc).__name__,
                  "error": str(exc), "finished_utc_epoch": time.time()}))
        raise
    else:
        files = {p.name: file_sha(p) for p in directory.iterdir()
                 if p.suffix in (".json", ".jsonl")}
        immutable(directory / "complete.json", seal({"stage": name, "files": files,
                                                    "finished_utc_epoch": time.time()}))


class Ledger:
    def __init__(self, directory):
        self.directory = inside(directory)
        self.db = sqlite3.connect(self.directory / "rollouts.sqlite")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE tasks (key TEXT PRIMARY KEY, status TEXT NOT NULL, data TEXT NOT NULL)")
        self.db.commit()
        self.used = 0

    def begin_many(self, items):
        self.db.executemany("INSERT INTO tasks VALUES (?, 'started', ?)",
                            [(k, canonical(m)) for k, m in items])
        self.db.commit()
        self.used += len(items)

    def finish_many(self, items):
        for key, status, data in items:
            if status not in ("ok", "failed"):
                raise ValueError("Invalid terminal status")
            cur = self.db.execute("UPDATE tasks SET status=?,data=? WHERE key=? AND status='started'",
                                  (status, canonical(data), key))
            if cur.rowcount != 1:
                raise RuntimeError("Only one transition from started is allowed")
        self.db.commit()

    def mark_invoked(self, items):
        for key, data in items:
            cur = self.db.execute("UPDATE tasks SET data=? WHERE key=? AND status='started'",
                                  (canonical({**data, "repair_invoked": True}), key))
            if cur.rowcount != 1:
                raise RuntimeError("Cannot invoke completed task")
        self.db.commit()

    def fail_started(self, items):
        self.db.rollback()
        for key, data in items:
            self.db.execute("UPDATE tasks SET status='failed',data=? WHERE key=? AND status='started'",
                            (canonical(data), key))
        self.db.commit()

    def export_close(self, extra):
        # Stream rows instead of materializing a million dictionaries in memory.
        costs = {"attempt_records": 0, "completed": 0, "failed": 0, "uncertain": 0,
                 "repair_invocations_known": 0, "uncertain_invocation_upper_bound": 0}
        path = self.directory / "rows.jsonl"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as f:
                for key, status, data in self.db.execute("SELECT key,status,data FROM tasks ORDER BY key"):
                    row = {"key": key, "status": status, **__import__("json").loads(data)}
                    costs["attempt_records"] += 1
                    costs[{"ok": "completed", "failed": "failed", "started": "uncertain"}[status]] += 1
                    costs["repair_invocations_known"] += row.get("repair_invoked") is True
                    costs["uncertain_invocation_upper_bound"] += status == "started"
                    f.write(canonical(row) + "\n")
                f.flush(); os.fsync(f.fileno())
            immutable(self.directory / "export.json", seal({"rows_file": "rows.jsonl",
                      "rows_sha256": file_sha(path), "costs": costs, **extra,
                      "count_caveat": "durably issued rollouts; an interrupted backend has uncertain completion, never zero cost"}))
        finally:
            self.db.close()


def export_rows(directory, export):
    directory = inside(directory)
    if export.get("rows_file") != "rows.jsonl":
        raise ValueError("Unexpected export filename")
    path = inside(directory / "rows.jsonl")
    if file_sha(path) != export["rows_sha256"]:
        raise RuntimeError("Rollout export changed")
    import json
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
