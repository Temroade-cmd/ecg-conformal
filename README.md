# ecg-conformal

Projet personnel d'apprentissage. Je le construis étape par étape pour apprendre tout le chemin d'un projet d'IA en santé, des données brutes jusqu'à une app déployée en ligne.

Le but final est un classifieur d'ECG 12 dérivations, entraîné sur [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/), qui sait dire quand il ne sait pas (prédiction conforme).

Chaque étape est expliquée dans [JOURNAL.md](JOURNAL.md), avec le pourquoi de chaque outil.

## Avancement

- [x] 1. Squelette du projet et chargement de PTB-XL
- [x] 2. Le modèle
- [ ] 3. L'entraînement
- [ ] 4. Le vrai entraînement sur GitHub Actions
- [ ] 5. La prédiction conforme
- [ ] 6. L'évaluation
- [ ] 7. L'app Streamlit et le déploiement
- [ ] 8. README final et figures

## Lancer les tests

```bash
pip install -e ".[dev]"
pytest -q
```

Données PTB-XL, PhysioNet, licence CC BY 4.0.
