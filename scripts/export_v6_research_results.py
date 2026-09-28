"""Export the compact, non-market-data V6 audit bundle for version control."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "v6_development"
DESTINATION = ROOT / "research_results" / "v6"
FILES = ("metrics.json", "training_audit.csv", "model_leaderboard.csv", "policy_leaderboard.csv")


def main() -> None:
    missing = [name for name in FILES if not (SOURCE / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing V6 artifacts: {missing}")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copy2(SOURCE / name, DESTINATION / name)
    print(f"Exported {len(FILES)} V6 research files to {DESTINATION}")


if __name__ == "__main__":
    main()
