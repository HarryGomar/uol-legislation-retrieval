"""Run a Round 3 module as the only heavy Round 3 process on the machine (shared-RAM agreement, 2026-10-14).

    python -m uol.round3.heavy <module> [args...]      # e.g. python -m uol.round3.heavy uol.round3.s3_cocitation run --nulls 100

Takes ``.cache/round3_heavy.lock`` (owner PID inside; reclaimed when the owner has died), then runs the module's
``__main__`` in this process with ``runpy``, so the whole process lifetime, feature builds included, is covered.
``--wait-pid N`` first waits for an existing process (started before this lock existed) to exit.
"""
from __future__ import annotations

import atexit
import os
import runpy
import sys
import time
from pathlib import Path

from uol.embedding_push import common as ep

LOCK = Path(__file__).resolve().parents[3] / ".cache" / "round3_heavy.lock"


PAUSE = LOCK.with_suffix(".PAUSE")          # present = no new heavy Round 3 process may start (author/Coordinator pause)


def acquire(poll: int = 20) -> None:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    while True:
        if PAUSE.exists():                    # re-checked before every attempt: a pause holds waiters too
            time.sleep(poll)
            continue
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            if PAUSE.exists():                # a pause set during the attempt wins
                LOCK.unlink(missing_ok=True)
                continue
            atexit.register(lambda: LOCK.unlink(missing_ok=True) if LOCK.exists() and LOCK.read_text() == str(os.getpid()) else None)
            return
        except FileExistsError:
            try:
                owner = int(LOCK.read_text() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and not ep.pid_alive(owner):
                LOCK.unlink(missing_ok=True)
                continue
            time.sleep(poll)


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--wait-pid"]:
        pid = int(args[1])
        args = args[2:]
        while ep.pid_alive(pid):
            time.sleep(30)
    module = args[0]
    acquire()
    print(f"[heavy] {module} {' '.join(args[1:])} holds {LOCK} (pid {os.getpid()})", flush=True)
    sys.argv = [module] + args[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
