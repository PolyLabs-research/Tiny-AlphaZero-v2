"""Run an xboard protocol-1 engine behind a protocol-2 face, for python-chess (PREREG-p2_y).

    .venv/bin/python scripts/xboard1_adapter.py [--name NAME] -- ENGINE [ARGS...]

python-chess talks to this process as an xboard engine; it starts ENGINE and relays
between them through ``tinyalphazero.xboard1.Relay``, which documents the four things it
changes. The engine's own chess is untouched. When
python-chess quits or closes the pipe, the engine's stdin closes and it exits.
"""

from __future__ import annotations

import argparse
import contextlib
import subprocess
import sys
import threading

from tinyalphazero.xboard1 import Relay


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--name", default="micro-Max 4.8", help="myname declared to the GUI")
    ap.add_argument("engine", nargs=argparse.REMAINDER, help="-- ENGINE [ARGS...]")
    args = ap.parse_args(argv)
    cmd = args.engine[1:] if args.engine[:1] == ["--"] else args.engine
    if not cmd:
        ap.error("no engine command after --")

    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
    )
    out_lock = threading.Lock()
    relay = Relay(args.name)

    def to_gui(lines: list[str]) -> None:
        if lines:
            with out_lock:
                sys.stdout.write("".join(line + "\n" for line in lines))
                sys.stdout.flush()

    def relay_engine() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            to_gui(relay.from_engine(line))

    reader = threading.Thread(target=relay_engine, daemon=True)
    reader.start()

    assert proc.stdin is not None
    try:
        for line in sys.stdin:
            to_engine, back = relay.from_gui(line)
            to_gui(back)
            for out in to_engine:
                proc.stdin.write(out + "\n")
            proc.stdin.flush()
            if line.strip() == "quit":
                break
    except BrokenPipeError:
        pass
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        reader.join(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
