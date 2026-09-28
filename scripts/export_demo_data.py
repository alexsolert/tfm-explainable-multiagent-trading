"""Exporta un conjunto de resultados seguro y ligero para la demo pública."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts"
DESTINATION = ROOT / "demo_data"

FILES = (
    "walk_forward/validation_metrics.json",
    "walk_forward/validation_decisions.csv",
    "walk_forward/validation_equity.csv",
    "explainability/lime_cases.json",
    "hybrid_validation/metrics.json",
    "hybrid_validation/decisions.csv",
    "hybrid_validation/strategy.csv",
    "final_quantitative/final_test_metrics.json",
    "final_quantitative/final_test_decisions.csv",
    "final_quantitative/final_test_equity.csv",
    "hybrid_final_test/metrics.json",
    "hybrid_final_test/decisions.csv",
    "hybrid_final_test/strategy.csv",
    "v2_development/metrics.json",
    "v2_development/decisions.csv",
    "v2_development/equity.csv",
    "v2_development/model_leaderboard.csv",
    "v2_protected_test/metrics.json",
    "v2_protected_test/decisions.csv",
    "v2_protected_test/equity.csv",
    "v3_development/metrics.json",
    "v3_development/decisions.csv",
    "v3_development/equity.csv",
)


def main() -> None:
    missing = [relative for relative in FILES if not (SOURCE / relative).exists()]
    if missing:
        formatted = "\n".join(f"- {relative}" for relative in missing)
        raise FileNotFoundError(f"Faltan artefactos para la demo:\n{formatted}")

    for relative in FILES:
        target = DESTINATION / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, target)

    print(f"Demo exportada en {DESTINATION} ({len(FILES)} archivos).")


if __name__ == "__main__":
    main()
