"""Écrit un petit résumé de l'entraînement en markdown, à partir de history.json.

    python scripts/summary.py runs/main/history.json
    python scripts/summary.py runs/main/history.json runs/main/results.json

Avec results.json, le résumé ajoute les scores sur le test.

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


def test_summary(res, alpha="0.1"):
    # les scores sur le test, une ligne par diagnostic
    m = res["macro_auroc"]
    lines = [
        "## Test",
        "",
        f"Test macro AUROC {m['value']:.4f} (IC 95% {m['ci95'][0]:.4f} à {m['ci95'][1]:.4f}), {res['n_test']} patients",
        "",
        f"Prédiction conforme avec alpha = {alpha}, la couverture attendue est d'au moins {1 - float(alpha):.0%}.",
        "",
    ]
    for name, r in res["mondrian"][alpha].items():
        lines.append(
            f"- {name}, AUROC {res['auroc_per_class'][name]:.3f}, couverture malades {r['coverage_pos']:.1%}, "
            f"non malades {r['coverage_neg']:.1%}, tranché {r['decided_rate']:.1%}, je ne sais pas {r['uncertain_rate']:.1%}"
        )
    c = res["crc"][alpha]
    lines += [
        "",
        f"Contrôle du risque, seuil {c['threshold']:.3f}, diagnostics oubliés {c['fnr']:.1%}, "
        f"{c['mean_set_size']:.2f} diagnostics annoncés par patient pour {c['mean_true_labels']:.2f} vrais",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    history = json.loads(Path(sys.argv[1]).read_text())
    print(summary(history))
    if len(sys.argv) > 2:
        print(test_summary(json.loads(Path(sys.argv[2]).read_text())))
