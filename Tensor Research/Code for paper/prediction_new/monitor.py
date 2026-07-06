"""prediction_new/monitor.py

Snapshot the four Optuna studies: trial counts, best CV R², best params.
Run anytime; reads the shared journal files (no lock contention with workers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import optuna  # noqa: E402
from optuna.storages import JournalStorage  # noqa: E402
from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock  # noqa: E402

from prediction_config import LOOKBACKS, MODES, journal_path, study_name  # noqa: E402


def objective_journal_path(mode: str, L: int, objective_name: str) -> Path:
    base = journal_path(mode, L)
    if objective_name == "pooled_r2":
        return base
    return base.with_name(f"{base.stem}_{objective_name}{base.suffix}")


def objective_study_name(mode: str, L: int, objective_name: str) -> str:
    base = study_name(mode, L)
    if objective_name == "pooled_r2":
        return base
    return f"{base}_{objective_name}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--objective",
        default="pooled_r2",
        choices=[
            "pooled_r2",
            "residual_delta",
            "residual_delta_v2",
            "residual_delta_v3",
            "ridge_delta_v3",
        ],
    )
    args = parser.parse_args()

    residual_like = args.objective in (
        "residual_delta", "residual_delta_v2",
        "residual_delta_v3", "ridge_delta_v3",
    )
    score_label = "best_delta" if residual_like else "best_cv_r2"
    is_v2 = args.objective == "residual_delta_v2"
    is_v3 = args.objective in ("residual_delta_v3", "ridge_delta_v3")
    if is_v3:
        header = (
            f"{'study':<54}  {'n_trials':>8}  {score_label:>12}  {'rank':>4}  "
            f"{'reg_w':>12}  {'gamma':>6}  {'rms':>5}  {'fY':>3}  {'fX':>3}"
        )
    elif is_v2:
        header = (
            f"{'study':<54}  {'n_trials':>8}  {score_label:>12}  {'rank':>4}  "
            f"{'reg_w':>12}  {'gamma':>6}  {'rms':>5}  {'feat_sc':>7}"
        )
    else:
        header = (
            f"{'study':<54}  {'n_trials':>8}  {score_label:>12}  {'rank':>4}  "
            f"{'reg_w':>12}  {'rms':>5}"
        )
    print(header)
    print("-" * len(header))
    modes_iter = ("LEVELS",) if is_v3 else MODES
    for mode in modes_iter:
        for L in LOOKBACKS:
            jp = objective_journal_path(mode, L, args.objective)
            sn = objective_study_name(mode, L, args.objective)
            if not jp.exists():
                print(f"{sn:<48}  (no journal yet)")
                continue
            storage = JournalStorage(
                JournalFileBackend(str(jp), lock_obj=JournalFileSymlinkLock(str(jp)))
            )
            try:
                study = optuna.load_study(study_name=sn, storage=storage)
                n = len(study.trials)
                completed = [t for t in study.trials if t.value is not None]
                if not completed:
                    print(f"{sn:<48}  {n:>8}  (no completed yet)")
                    continue
                best = max(completed, key=lambda t: t.value)
                bp = best.params
                if is_v3:
                    print(
                        f"{sn:<54}  {n:>8}  {best.value:>12.5f}  "
                        f"{bp.get('RANK_REGRESS', '?'):>4}  "
                        f"{bp.get('REG_W', float('nan')):>12.4g}  "
                        f"{bp.get('GAMMA', float('nan')):>6.3f}  "
                        f"{str(bp.get('USE_RMS_SCALING', '?')):>5}  "
                        f"{str(bp.get('FEATURE_TARGET_SCALE', '?')):>3}  "
                        f"{str(bp.get('FEATURE_X_SCALE', '?')):>3}"
                    )
                elif is_v2:
                    print(
                        f"{sn:<54}  {n:>8}  {best.value:>12.5f}  "
                        f"{bp.get('RANK_REGRESS', '?'):>4}  "
                        f"{bp.get('REG_W', float('nan')):>12.4g}  "
                        f"{bp.get('GAMMA', float('nan')):>6.3f}  "
                        f"{str(bp.get('USE_RMS_SCALING', '?')):>5}  "
                        f"{str(bp.get('FEATURE_TARGET_SCALE', '?')):>7}"
                    )
                else:
                    print(
                        f"{sn:<54}  {n:>8}  {best.value:>12.5f}  "
                        f"{bp.get('RANK_REGRESS', '?'):>4}  "
                        f"{bp.get('REG_W', float('nan')):>12.4g}  "
                        f"{str(bp.get('USE_RMS_SCALING', '?')):>5}"
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"{sn:<48}  ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
