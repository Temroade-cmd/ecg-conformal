# Journal

Le pourquoi du comment, étape par étape. Ce que j'ai utilisé, et pourquoi ce choix plutôt qu'un autre.

---

## Étape 1. Squelette du projet et chargement de PTB-XL

### Ce qui a été ajouté

```
pyproject.toml            le projet Python et ses dépendances
src/ecgc/data.py          lecture de PTB-XL, labels, folds
scripts/make_fixture.py   un faux PTB-XL miniature pour les tests
tests/test_data.py        les tests de data.py
.github/workflows/ci.yml  les tests lancés automatiquement à chaque push
```

### pyproject.toml et le dossier src/

`pyproject.toml` est le fichier standard pour décrire un projet Python. On y met le nom, la version et les dépendances. Avec `pip install -e .` le code de `src/ecgc` devient importable partout (`from ecgc import data`), et le `-e` veut dire que les modifications du code sont prises en compte sans réinstaller.

Le code est dans `src/ecgc/` et pas directement à la racine. C'est la disposition "src layout". Elle évite un piège classique, les tests qui marchent par hasard parce qu'ils importent le dossier local au lieu du package installé.

### PTB-XL

21 799 ECG de 10 secondes, 12 dérivations, annotés par des cardiologues. Trois choses à lire.
- `ptbxl_database.csv`, une ligne par ECG, avec ses codes diagnostiques (`scp_codes`), son fold et le chemin du fichier signal
- `scp_statements.csv`, la table qui dit à quelle grande classe appartient chaque code
- `records100/`, les signaux à 100 Hz au format WFDB (un `.hea` qui décrit, un `.dat` qui contient les valeurs)

J'utilise la version 100 Hz et pas 500 Hz. Le benchmark de référence (Strodthoff et al. 2021) montre que ça suffit pour ces diagnostics, et c'est 5 fois moins lourd à entraîner.

### pandas et wfdb

`pandas` pour lire les deux CSV, c'est l'outil standard pour les tableaux. `wfdb` est la librairie officielle de PhysioNet pour lire leurs signaux, `wfdb.rdsamp` renvoie le signal et les infos (noms des dérivations, fréquence).

### Les 5 superclasses

Chaque ECG a plusieurs codes précis (IMI, LVH, NDT...). On les regroupe en 5 classes, normal, infarctus, anomalie ST/T, trouble de conduction, hypertrophie. Un ECG peut en avoir plusieurs, c'est du **multi-label**, donc `Y` est un tableau de 0 et de 1 avec 5 colonnes, pas une seule étiquette.

Comme dans le benchmark, tous les codes listés comptent, même ceux avec un score de confiance bas. Ça permet de comparer mes résultats aux leurs.

On enlève les ECG qui n'ont aucune des 5 classes, encore pour suivre le même protocole.

### Les folds

PTB-XL est déjà découpé en 10 folds, et **un même patient n'est jamais dans deux folds**. C'est essentiel. Si un patient était à la fois dans le train et le test, le modèle pourrait reconnaître le patient plutôt que la maladie, et les résultats seraient trop beaux.

| Folds | Rôle |
|---|---|
| 1 à 7 | entraînement |
| 8 | validation, choisir la meilleure époque |
| 9 | calibration conforme (étape 5) |
| 10 | test, utilisé une seule fois à la fin |

Pourquoi un fold séparé pour la calibration au lieu de réutiliser la validation ? Parce que la validation sert à choisir le modèle. Une fois qu'on a fait un choix avec des données, elles ne sont plus "neuves" pour ce modèle. La prédiction conforme a besoin de données neuves pour que sa garantie tienne. Les folds 9 et 10 ont en plus des labels vérifiés à la main, donc c'est la meilleure paire calibration et test.

### Le cache .npz

Lire 21 799 petits fichiers prend plusieurs minutes. On le fait une seule fois (`build_cache`) et on sauve tout dans un seul fichier `.npz` (le format compressé de numpy). Les signaux sont stockés en `float16` pour diviser la taille par deux, la précision reste largement suffisante pour des millivolts.

### Un piège rencontré

Dans les fichiers PTB-XL, les dérivations s'appellent `AVR`, `AVL`, `AVF` en majuscules, alors qu'on les écrit d'habitude `aVR`, `aVL`, `aVF`. Le code compare donc les noms en majuscules avant de remettre les dérivations dans l'ordre.

### Le faux dataset et pytest

Pour tester sans télécharger 1,7 Go, `make_fixture.py` crée quelques centaines de faux ECG avec **exactement la même structure de fichiers** que PTB-XL (mêmes CSV, mêmes fichiers WFDB, mêmes noms de dérivations en majuscules). Si le code lit correctement le faux, il lira correctement le vrai.

`pytest` lance toutes les fonctions qui commencent par `test_` dans le dossier `tests/`. Le dossier temporaire `tmp_path` est fourni par pytest et supprimé après.

### GitHub Actions

`ci.yml` dit à GitHub de lancer les tests sur une machine propre à chaque push. Si je casse quelque chose, je le vois tout de suite, et le badge vert prouve que le projet s'installe et marche ailleurs que sur mon ordinateur.
