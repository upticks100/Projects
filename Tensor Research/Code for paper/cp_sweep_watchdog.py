"""Lightweight RAM watchdog for long-running Python jobs.

Polls a target Python process (matched by a substring of its command line and
owning user) plus host system memory. By default, SIGTERMs (then SIGKILLs
after a grace period) the matched target if either:

  * the target process's RSS exceeds --process-pct of total system RAM, or
  * the host's "used" RAM exceeds --system-limit-gb.

With --cascade, on a system-RAM breach the watchdog instead enumerates all
python processes owned by --user, sorts them by RSS descending, and kills
them in order until system used RAM falls below
    --system-limit-gb minus --cleanup-margin-gb
or no candidates remain. Use --cascade-match SUBSTR to restrict the *non
target* candidate set to processes whose cmdline contains SUBSTR (useful if
the user owns unrelated python processes such as IDE language servers). The
matched target is always included as a cascade candidate regardless of
--cascade-match.

All measurements are taken via direct reads from /proc, so polling at 5 s is
essentially free (~5-15 ms of work per cycle, ~20 MB resident).

Examples
--------
Watch a specific script with 5 s polling and tmux teardown:

    python cp_sweep_watchdog.py \\
        --match "Build_PrePrediction_Exhibits.py" \\
        --interval 5 \\
        --system-limit-gb 56 \\
        --log "pre_prediction_cache/cp_sweep_watchdog.log" \\
        --tmux-session cp-sweep-raw

Cascade mode that kills any of mcnama53's python jobs whose cmdline contains
"Tensor Research" until system RAM is back under the threshold:

    python cp_sweep_watchdog.py \\
        --match "Build_PrePrediction_Exhibits.py" \\
        --interval 5 \\
        --system-limit-gb 56 \\
        --cascade \\
        --cascade-match "Tensor Research"
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pwd
import signal
import subprocess
import sys
import time
from pathlib import Path


_TOTAL_RAM_KB: int | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watchdog that terminates a Python job (or, in --cascade mode, any "
            "of the user's python jobs in RSS order) when configured RAM "
            "limits are crossed."
        ),
    )
    parser.add_argument(
        "--match",
        required=True,
        help="Substring that must appear in the target process's full command line.",
    )
    parser.add_argument(
        "--user",
        default=pwd.getpwuid(os.getuid()).pw_name,
        help="Only watch processes owned by this user (default: current user).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Seconds between polls (default: 5).",
    )
    parser.add_argument(
        "--system-limit-gb",
        type=float,
        default=56.0,
        help="Kill target when host 'used' RAM crosses this many GB (default: 56).",
    )
    parser.add_argument(
        "--process-pct",
        type=float,
        default=90.0,
        help="Kill target when its RSS is this percent of MemTotal (default: 90).",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append log lines to this file (also echoed to stdout).",
    )
    parser.add_argument(
        "--tmux-session",
        default=None,
        help=(
            "If set, run `tmux kill-session -t NAME` after the target dies "
            "(either directly or via cascade)."
        ),
    )
    parser.add_argument(
        "--grace-seconds",
        type=float,
        default=30.0,
        help="Seconds to wait between SIGTERM and SIGKILL (default: 30).",
    )
    parser.add_argument(
        "--start-timeout",
        type=float,
        default=120.0,
        help=(
            "If the target hasn't been seen yet, keep polling up to this many "
            "seconds before giving up (default: 120). Set to 0 to exit "
            "immediately if not found on the first poll."
        ),
    )
    parser.add_argument(
        "--comm-prefix",
        default="python",
        help=(
            "Only consider processes whose /proc/<pid>/comm starts with this "
            "prefix (default: python)."
        ),
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help=(
            "On a system-RAM breach, kill the user's python processes in RSS "
            "order until system used RAM is back below "
            "system_limit_gb - cleanup_margin_gb. Without --cascade only the "
            "matched target is killed."
        ),
    )
    parser.add_argument(
        "--cascade-match",
        default=None,
        help=(
            "When --cascade is set, restricts which *other* python processes "
            "(besides the matched target) can be killed by the cascade: only "
            "processes whose cmdline contains this substring are eligible. "
            "Use this to avoid killing unrelated python processes such as IDE "
            "language servers or jupyter. The matched target is ALWAYS a "
            "cascade candidate regardless of this filter. Default: no filter "
            "(any python process owned by --user is eligible)."
        ),
    )
    parser.add_argument(
        "--cleanup-margin-gb",
        type=float,
        default=2.0,
        help=(
            "Cascade keeps killing until system used RAM is at least this "
            "many GB below --system-limit-gb (default: 2)."
        ),
    )
    return parser.parse_args(argv)


def total_ram_kb() -> int:
    global _TOTAL_RAM_KB
    if _TOTAL_RAM_KB is None:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    _TOTAL_RAM_KB = int(line.split()[1])
                    break
        if _TOTAL_RAM_KB is None:
            raise RuntimeError("Could not read MemTotal from /proc/meminfo")
    return _TOTAL_RAM_KB


def system_used_gb() -> float:
    """Return the same 'used' figure that `free` reports, in GB."""
    info: dict[str, int] = {}
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            key, _, rest = line.partition(":")
            parts = rest.strip().split()
            if parts:
                info[key] = int(parts[0])
    total = info.get("MemTotal", 0)
    free = info.get("MemFree", 0)
    buffers = info.get("Buffers", 0)
    cached = info.get("Cached", 0) + info.get("SReclaimable", 0) - info.get("Shmem", 0)
    used_kb = total - free - buffers - cached
    return used_kb / 1024 / 1024


def _read_status(pid: int) -> dict[str, str] | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as handle:
            return {
                key: rest.strip()
                for key, _, rest in (line.partition(":") for line in handle)
                if key
            }
    except OSError:
        return None


def _read_cmdline(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as handle:
            raw = handle.read()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode(errors="ignore")


def _read_comm(pid: int) -> str | None:
    try:
        with open(f"/proc/{pid}/comm", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def proc_rss_kb(pid: int) -> int | None:
    status = _read_status(pid)
    if status is None or "VmRSS" not in status:
        return None
    return int(status["VmRSS"].split()[0])


def proc_uid(pid: int) -> int | None:
    status = _read_status(pid)
    if status is None or "Uid" not in status:
        return None
    return int(status["Uid"].split()[0])


def find_target_pid(match: str, user_uid: int, comm_prefix: str) -> int | None:
    self_pid = os.getpid()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == self_pid:
            continue
        comm = _read_comm(pid)
        if comm is None:
            continue
        if comm_prefix and not comm.startswith(comm_prefix):
            continue
        cmdline = _read_cmdline(pid)
        if cmdline is None or match not in cmdline:
            continue
        uid = proc_uid(pid)
        if uid is None or uid != user_uid:
            continue
        return pid
    return None


def list_candidates(
    user_uid: int,
    comm_prefix: str,
    cmdline_substr: str | None,
    exclude_pids: set[int] | None = None,
) -> list[tuple[int, str, int, str]]:
    """Return [(pid, comm, rss_kb, cmdline)] for matching live processes,
    sorted by rss_kb descending."""
    self_pid = os.getpid()
    exclude_pids = exclude_pids or set()
    out: list[tuple[int, str, int, str]] = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        pid = int(name)
        if pid == self_pid or pid in exclude_pids:
            continue
        comm = _read_comm(pid)
        if comm is None:
            continue
        if comm_prefix and not comm.startswith(comm_prefix):
            continue
        status = _read_status(pid)
        if status is None or "Uid" not in status:
            continue
        try:
            uid = int(status["Uid"].split()[0])
        except ValueError:
            continue
        if uid != user_uid:
            continue
        cmdline = _read_cmdline(pid) or ""
        if cmdline_substr is not None and cmdline_substr not in cmdline:
            continue
        try:
            rss_kb = int(status["VmRSS"].split()[0]) if "VmRSS" in status else 0
        except ValueError:
            rss_kb = 0
        out.append((pid, comm, rss_kb, cmdline))
    out.sort(key=lambda r: r[2], reverse=True)
    return out


class Logger:
    def __init__(self, path: Path | None) -> None:
        self.path = path
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{dt.datetime.now().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        if self.path is not None:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _wait_for_exit(pid: int, grace_seconds: float) -> bool:
    deadline = time.time() + grace_seconds
    while time.time() < deadline:
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.5)
    return not os.path.exists(f"/proc/{pid}")


def stop_pid(pid: int, log: Logger, grace_seconds: float) -> bool:
    """SIGTERM, then SIGKILL after grace_seconds. Returns True if killed."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        log(f"pid={pid} already gone before SIGTERM")
        return True
    if _wait_for_exit(pid, grace_seconds):
        return True
    log(f"pid={pid} still alive after {grace_seconds:.0f}s; sending SIGKILL")
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    return _wait_for_exit(pid, 5.0)


def stop_target(
    pid: int,
    reason: str,
    log: Logger,
    grace_seconds: float,
    tmux_session: str | None,
) -> None:
    log(f"{reason}; sending SIGTERM to pid={pid}")
    stop_pid(pid, log, grace_seconds)
    if tmux_session:
        subprocess.run(["tmux", "kill-session", "-t", tmux_session], check=False)


def _target_candidate(
    target_pid: int | None,
    user_uid: int,
) -> tuple[int, str, int, str] | None:
    """Build a candidate tuple for the target PID, ignoring cmdline filters.

    Still verifies the PID is alive and owned by the expected user.
    """
    if target_pid is None:
        return None
    status = _read_status(target_pid)
    if status is None or "Uid" not in status:
        return None
    try:
        uid = int(status["Uid"].split()[0])
    except ValueError:
        return None
    if uid != user_uid:
        return None
    try:
        rss_kb = int(status["VmRSS"].split()[0]) if "VmRSS" in status else 0
    except ValueError:
        rss_kb = 0
    comm = _read_comm(target_pid) or "?"
    cmdline = _read_cmdline(target_pid) or ""
    return target_pid, comm, rss_kb, cmdline


def cascade_kill(
    log: Logger,
    user_uid: int,
    comm_prefix: str,
    cmdline_substr: str | None,
    target_used_gb: float,
    grace_seconds: float,
    target_pid: int | None,
) -> set[int]:
    """Kill candidates by RSS desc until system_used <= target_used_gb.

    The target PID, if given and still alive, is ALWAYS treated as a
    candidate regardless of cmdline_substr.
    """
    killed: set[int] = set()
    while True:
        sys_used = system_used_gb()
        if sys_used <= target_used_gb:
            log(
                f"cascade: system_used={sys_used:.2f}GB <= "
                f"target={target_used_gb:.2f}GB; stopping"
            )
            return killed
        candidates = list_candidates(
            user_uid, comm_prefix, cmdline_substr, exclude_pids=killed
        )
        if (
            target_pid is not None
            and target_pid not in killed
            and not any(c[0] == target_pid for c in candidates)
        ):
            target_row = _target_candidate(target_pid, user_uid)
            if target_row is not None:
                candidates.append(target_row)
                candidates.sort(key=lambda r: r[2], reverse=True)
        if not candidates:
            log(
                f"cascade: no more candidates but system_used={sys_used:.2f}GB > "
                f"target={target_used_gb:.2f}GB; stopping"
            )
            return killed
        pid, comm, rss_kb, _cmd = candidates[0]
        rss_gb = rss_kb / 1024 / 1024
        is_target = target_pid is not None and pid == target_pid
        log(
            f"cascade: SIGTERM pid={pid} comm={comm} rss={rss_gb:.2f}GB "
            f"system_used={sys_used:.2f}GB target={target_used_gb:.2f}GB"
            + (" (matched target)" if is_target else "")
        )
        stop_pid(pid, log, grace_seconds)
        killed.add(pid)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        user_uid = pwd.getpwnam(args.user).pw_uid
    except KeyError:
        print(f"Unknown user: {args.user}", file=sys.stderr)
        return 2

    log = Logger(args.log)
    total_gb = total_ram_kb() / 1024 / 1024
    log(
        "watchdog started "
        f"match={args.match!r} user={args.user}(uid={user_uid}) "
        f"interval={args.interval:.0f}s "
        f"system_used_limit={args.system_limit_gb:.1f}GB "
        f"process_pct_limit={args.process_pct:.1f}% "
        f"cascade={args.cascade} "
        f"cascade_match={args.cascade_match!r} "
        f"MemTotal={total_gb:.2f}GB"
    )

    seen_target = False
    waited = 0.0
    while True:
        pid = find_target_pid(args.match, user_uid, args.comm_prefix)

        if pid is None:
            if not seen_target and waited < args.start_timeout:
                log(f"target not found yet; waiting (waited={waited:.0f}s)")
                time.sleep(args.interval)
                waited += args.interval
                continue
            if not seen_target:
                log(
                    "target never appeared within "
                    f"start_timeout={args.start_timeout:.0f}s; exiting"
                )
            else:
                log("target process gone; exiting")
            return 0

        seen_target = True
        rss_kb = proc_rss_kb(pid)
        if rss_kb is None:
            log(f"pid={pid} disappeared while reading RSS; will retry")
            time.sleep(args.interval)
            continue
        rss_gb = rss_kb / 1024 / 1024
        process_pct = 100.0 * rss_kb / total_ram_kb()
        sys_used_gb = system_used_gb()

        log(
            f"pid={pid} process_ram={process_pct:.1f}% "
            f"rss={rss_gb:.2f}GB system_used={sys_used_gb:.2f}GB"
        )

        if process_pct >= args.process_pct:
            stop_target(
                pid,
                f"process RAM crossed {args.process_pct:.1f}%",
                log,
                args.grace_seconds,
                args.tmux_session,
            )
            return 0

        if sys_used_gb >= args.system_limit_gb:
            if not args.cascade:
                stop_target(
                    pid,
                    f"system used RAM crossed {args.system_limit_gb:.1f}GB",
                    log,
                    args.grace_seconds,
                    args.tmux_session,
                )
                return 0
            target_used_gb = max(
                0.0, args.system_limit_gb - args.cleanup_margin_gb
            )
            log(
                f"system used RAM crossed {args.system_limit_gb:.1f}GB; "
                f"cascading kills until <= {target_used_gb:.2f}GB"
            )
            killed = cascade_kill(
                log,
                user_uid,
                args.comm_prefix,
                args.cascade_match,
                target_used_gb,
                args.grace_seconds,
                target_pid=pid,
            )
            if pid in killed:
                log(f"cascade: target pid={pid} was killed; exiting")
                if args.tmux_session:
                    subprocess.run(
                        ["tmux", "kill-session", "-t", args.tmux_session],
                        check=False,
                    )
                return 0
            log(
                "cascade: target survived; resuming polling "
                f"({len(killed)} other process(es) terminated)"
            )

        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
