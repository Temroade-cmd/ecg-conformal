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

---

## Étape 2. Le modèle

### Ce que j'ai ajouté, dans l'ordre des commits

1. PyTorch dans les dépendances
2. Le bloc résiduel
3. Le modèle complet
4. Les tests du modèle, qui ont échoué
5. La correction du raccourci, les tests repassent au vert
6. La version déployable du modèle et son test

### Pourquoi PyTorch

C'est la librairie de deep learning la plus utilisée en recherche, et celle de la plupart des papiers sur PTB-XL. Dans la CI j'installe la version CPU (`--index-url .../whl/cpu`). La version normale embarque CUDA pour les cartes graphiques, plusieurs Go inutiles sur une machine GitHub qui n'a pas de GPU.

### Pourquoi un réseau convolutif 1D

Un ECG n'est pas une image, c'est 12 courbes dans le temps. Une convolution 1D fait glisser un petit filtre le long du temps, sur les 12 dérivations à la fois. Un filtre peut apprendre à reconnaître une forme locale, un QRS large, une onde T inversée, un sus-décalage du segment ST. Le même filtre sert à tous les endroits du signal, donc le modèle reconnaît un battement anormal qu'il soit au début ou à la fin des 10 secondes.

### Le bloc résiduel

Chaque bloc enchaîne 3 convolutions, de noyaux 7, 5 puis 3. À 100 Hz, un noyau de 7 couvre 70 ms, à peu près la durée d'un QRS. Après chaque convolution il y a une BatchNorm, qui remet les valeurs à une échelle stable et rend l'entraînement plus facile, et une ReLU, qui garde ce qui est positif et met le reste à zéro.

Quelques détails.
- `padding=k // 2` ajoute des zéros aux bords pour que le signal garde sa longueur de 1000 points.
- `bias=False` dans les convolutions, parce que la BatchNorm qui suit a déjà son propre décalage. Un biais avant serait annulé.
- Pas de ReLU après la 3e convolution. Elle vient après l'addition avec le raccourci.

Le **raccourci** (skip connection) est l'idée des ResNet (He et al., 2016). La sortie du bloc est `transformation(x) + x`. Le bloc n'apprend que ce qu'il faut ajouter à l'entrée, et pendant l'entraînement le gradient peut remonter directement par le raccourci. Sans ça, les réseaux profonds apprennent mal.

### L'erreur que j'ai faite

J'ai d'abord écrit le raccourci comme une simple addition, `body(x) + x`. Le test `test_block_keeps_length` passait, parce qu'il utilise un bloc de 64 vers 64 canaux. Mais les deux autres tests ont échoué dans la CI (le commit "Tests du modèle" est rouge sur GitHub).

La raison, le premier bloc reçoit les 12 dérivations et sort 64 canaux. On essaie donc d'additionner un tenseur de 64 canaux avec un tenseur de 12 canaux, et PyTorch refuse.

La correction, quand le nombre de canaux change, le raccourci passe par une convolution 1x1 (plus une BatchNorm). Une conv 1x1 ne regarde aucun voisin dans le temps, elle fait juste une combinaison des 12 canaux pour en fabriquer 64. Quand le nombre de canaux ne change pas, le raccourci reste l'identité. C'est exactement ce que fait le ResNet d'origine.

Ce que j'en retiens, un test qui passe sur un cas facile (64 vers 64) ne prouve pas que le cas réel (12 vers 64) marche. Il faut tester avec les vraies dimensions.

### Le modèle complet

Trois blocs, 12 → 64 → 128 → 128 canaux. Ensuite il faut passer d'un signal de 1000 points à un seul vecteur par ECG. Je fais deux résumés et je les colle.
- la **moyenne** sur le temps, ce qui est présent en général dans le tracé
- le **maximum** sur le temps, ce qui apparaît fortement au moins une fois, même sur un seul battement

Ça donne 256 valeurs, puis un dropout de 30% (pendant l'entraînement on éteint au hasard 30% des valeurs, pour éviter que le modèle compte trop sur quelques-unes), puis une couche linéaire vers 5 scores, un par diagnostic.

Au total environ 485 000 paramètres. C'est petit, ça s'entraîne sur CPU, et c'est proche du resnet1d_wang du benchmark PTB-XL.

Le modèle renvoie des **logits** (des scores entre moins l'infini et plus l'infini), pas encore des probabilités. La sigmoïde viendra dans la fonction de perte à l'étape 3, c'est plus stable numériquement.

### La version déployable

Pour l'app, je veux pouvoir donner un ECG brut en millivolts et recevoir 5 probabilités. `Deployable` enveloppe le modèle et fait les deux étapes autour, la normalisation avant et la sigmoïde après. La moyenne et l'écart-type sont enregistrés avec `register_buffer`, ils sont sauvegardés avec le modèle mais ne sont jamais modifiés par l'entraînement.

### Les tests

- La longueur du signal est conservée dans un bloc.
- Le modèle sort bien 4 × 5 scores pour 4 ECG.
- Le gradient arrive jusqu'à la toute première convolution, donc tout le réseau peut apprendre.
- `Deployable` donne exactement le même résultat que les étapes faites à la main, et des valeurs entre 0 et 1.
