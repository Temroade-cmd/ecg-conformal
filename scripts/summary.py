"""Écrit un petit résumé de l'entraînement en markdown, à partir de history.json.

    python scripts/summary.py runs/main/history.json

Sur GitHub Actions, la sortie est envoyée dans $GITHUB_STEP_SUMMARY,
ce qui l'affiche directement sur la page du run.
"""

import json
import sys
from pathlib import Path


def summary(history):
    # la meilleure époque est celle avec la plus haute AUROC de validation, c'est celle que fit() garde
    best = max(history, key=lambda h: h["val_macro_auc"])
    lines = [
        "## Entraînement",
        "",
        f"Meilleure époque {best['epoch']}, AUROC de validation {best['val_macro_auc']:.4f}",
        "",
        "| Époque | Perte (train) | AUROC (validation) |",
        "|---:|---:|---:|",
    ]
    for h in history:
        row = f"| {h['epoch']} | {h['train_loss']:.4f} | {h['val_macro_auc']:.4f} |"
        if h is best:
            row = f"| **{h['epoch']}** | **{h['train_loss']:.4f}** | **{h['val_macro_auc']:.4f}** |"  # en gras
        lines.append(row)
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    history = json.loads(Path(sys.argv[1]).read_text())
    print(summary(history))
