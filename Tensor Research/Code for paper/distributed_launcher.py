"""Generic SSH/NFS launcher for lab-machine sweeps.

This script launches arbitrary commands across lab computers while also starting
a host-local RAM watchdog for the launched process family. It assumes:

1. The project directory is shared across hosts (NFS home).
2. SSH access has been opened manually or key auth is available.
3. Each job command writes its own result file or is otherwise resume-safe.

Examples
--------
Launch one command on two hosts:

    python distributed_launcher.py \\
        --hosts utmlab20-02,utmlab20-15 \\
        --cwd "/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new" \\
        --job-name imputer-sensitivity \\
        --watchdog-match "imputer_sensitivity.py" \\
        --command 'OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 /student/mcnama53/.local/share/mamba/envs/research/bin/python -u imputer_sensitivity.py --configs validated_cv --modes LEVELS --lookbacks 2 --output results/validated_levels_L2.csv'

Launch a job list, round-robin across hosts:

    python distributed_launcher.py \\
        --hosts utmlab20-02,utmlab20-15,utmlab26-17 \\
        --cwd "/student/mcnama53/Projects/Tensor Research/Code for paper/prediction_new" \\
        --job-name prediction-sensitivity \\
        --watchdog-match "imputer_sensitivity.py" \\
        --job-file jobs/imputer_sensitivity_jobs.txt
"""

from __future__ import annotations

import argparse
import math
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PYTHON = "/student/mcnama53/.local/share/mamba/envs/research/bin/python"
DEFAULT_PROJECT = "/student/mcnama53/Projects/Tensor Research/Code for paper"


@dataclass(frozen=True)
class Job:
    idx: int
    host: str
    command: str


def parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_local(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def run_ssh(host: str, remote_command: str, *, check: bool = True) -> subprocess.CompletedProcess:
    return run_local(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            remote_command,
        ],
        check=check,
    )


def remote_memtotal_gb(host: str) -> float:
    proc = run_ssh(host, "awk '/MemTotal:/ {print $2}' /proc/meminfo")
    kb = int(proc.stdout.strip())
    return kb / 1024 / 1024


def remote_hostname(host: str) -> str:
    proc = run_ssh(host, "hostname")
    return proc.stdout.strip()


def build_jobs(hosts: list[str], command: str | None, job_file: Path | None) -> list[Job]:
    if command and job_file:
        raise ValueError("Use either --command or --job-file, not both.")
    if not command and not job_file:
        raise ValueError("Provide --command or --job-file.")

    if command:
        commands = [command for _ in hosts]
    else:
        commands = [
            line.strip()
            for line in job_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

    jobs = []
    for idx, cmd in enumerate(commands):
        jobs.append(Job(idx=idx, host=hosts[idx % len(hosts)], command=cmd))
    return jobs


def quote_path(path: str | Path) -> str:
    return shlex.quote(str(path))


def start_watchdog(
    host: str,
    cwd: str,
    job_name: str,
    watchdog_match: str,
    ram_pct: float,
    cleanup_margin_gb: float,
    log_dir: Path,
    interval: int,
    start_timeout: int,
    dry_run: bool,
) -> None:
    total_gb = remote_memtotal_gb(host)
    limit_gb = math.floor(total_gb * ram_pct / 100.0 * 10) / 10
    host_label = remote_hostname(host)
    log_path = log_dir / f"{job_name}_{host_label}_watchdog.log"
    watchdog = (
        f"cd {quote_path(cwd)} && nohup {quote_path(DEFAULT_PYTHON)} -u "
        f"{quote_path(Path(DEFAULT_PROJECT) / 'cp_sweep_watchdog.py')} "
        f"--match {shlex.quote(watchdog_match)} "
        f"--interval {interval} "
        f"--system-limit-gb {limit_gb:.1f} "
        f"--process-pct 85 "
        f"--cascade "
        f"--cascade-match {shlex.quote(watchdog_match)} "
        f"--cleanup-margin-gb {cleanup_margin_gb:.1f} "
        f"--start-timeout {start_timeout} "
        f"--log {quote_path(log_path)} "
        f"> {quote_path(log_path.with_suffix('.stdout.log'))} 2>&1 < /dev/null & "
        f"echo WATCHDOG_PID=$!"
    )
    print(
        f"[watchdog] host={host} mem={total_gb:.1f}GB limit={limit_gb:.1f}GB "
        f"log={log_path}"
    )
    if dry_run:
        print(f"  {watchdog}")
        return
    proc = run_ssh(host, watchdog)
    print(proc.stdout.strip())


def launch_job(
    job: Job,
    cwd: str,
    job_name: str,
    log_dir: Path,
    dry_run: bool,
) -> None:
    host_label = remote_hostname(job.host)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{job_name}_{job.idx:03d}_{host_label}_{timestamp}.log"
    remote = (
        f"cd {quote_path(cwd)} && nohup bash -lc {shlex.quote(job.command)} "
        f"> {quote_path(log_path)} 2>&1 < /dev/null & "
        f"echo JOB_PID=$! HOST=$(hostname) LOG={quote_path(log_path)}"
    )
    print(f"[job] idx={job.idx} host={job.host} log={log_path}")
    if dry_run:
        print(f"  {remote}")
        return
    proc = run_ssh(job.host, remote)
    print(proc.stdout.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hosts", required=True, help="Comma-separated SSH hosts/aliases.")
    parser.add_argument("--cwd", required=True, help="Remote working directory.")
    parser.add_argument("--job-name", required=True, help="Short name used for logs.")
    parser.add_argument(
        "--watchdog-match",
        required=True,
        help="Substring identifying our worker processes for watchdog cascade.",
    )
    parser.add_argument("--command", default=None, help="Command to run once per host.")
    parser.add_argument("--job-file", type=Path, default=None, help="One shell command per line.")
    parser.add_argument("--ram-pct", type=float, default=85.0)
    parser.add_argument("--cleanup-margin-gb", type=float, default=5.0)
    parser.add_argument("--watchdog-interval", type=int, default=10)
    parser.add_argument("--watchdog-start-timeout", type=int, default=600)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path(DEFAULT_PROJECT) / "distributed_logs",
    )
    parser.add_argument("--no-watchdog", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hosts = parse_csv(args.hosts)
    jobs = build_jobs(hosts, args.command, args.job_file)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"hosts={hosts}")
    print(f"jobs={len(jobs)} log_dir={args.log_dir}")

    unique_hosts = sorted({job.host for job in jobs})
    if not args.no_watchdog:
        for host in unique_hosts:
            start_watchdog(
                host=host,
                cwd=args.cwd,
                job_name=args.job_name,
                watchdog_match=args.watchdog_match,
                ram_pct=args.ram_pct,
                cleanup_margin_gb=args.cleanup_margin_gb,
                log_dir=args.log_dir,
                interval=args.watchdog_interval,
                start_timeout=args.watchdog_start_timeout,
                dry_run=args.dry_run,
            )

    for job in jobs:
        launch_job(
            job=job,
            cwd=args.cwd,
            job_name=args.job_name,
            log_dir=args.log_dir,
            dry_run=args.dry_run,
        )

    print("Launch complete.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print("Command failed:", " ".join(exc.cmd), file=sys.stderr)
        print(exc.stdout, file=sys.stderr)
        print(exc.stderr, file=sys.stderr)
        raise
