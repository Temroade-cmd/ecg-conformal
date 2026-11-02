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

---

## Étape 3. L'entraînement

### Ce que j'ai ajouté, dans l'ordre des commits

1. La métrique macro AUROC
2. L'augmentation des données
3. La fonction `predict`
4. La boucle d'entraînement `fit`
5. Les tests de l'entraînement, qui ont échoué
6. La correction multi-label, les tests repassent au vert
7. Le script `run.py`
8. Le test de la commande complète
9. La correction d'une fuite de données
10. La correction du choix de la meilleure époque

### Pourquoi l'AUROC

Pour chaque diagnostic, l'AUROC répond à une question simple. Si je prends au hasard un ECG qui a le diagnostic et un ECG qui ne l'a pas, quelle est la probabilité que le modèle donne un score plus haut au premier ? 0,5 c'est le hasard, 1 c'est parfait. L'avantage, ça ne dépend d'aucun seuil, et ça reste lisible même quand une classe est rare (l'hypertrophie l'est). La **macro** AUROC est la moyenne sur les 5 diagnostics, chacun compte autant. C'est aussi la métrique du benchmark PTB-XL, donc je peux comparer. J'utilise `roc_auc_score` de scikit-learn plutôt que de la recoder.

### L'augmentation des données

Pendant l'entraînement, chaque ECG est légèrement modifié à chaque passage, pour que le modèle ne retienne pas les exemples par cœur.
- **Amplitude** multipliée par un facteur entre 0,9 et 1,1, comme des électrodes plus ou moins bien posées.
- **Dérive de la ligne de base**, une lente ondulation ajoutée, comme quand le patient respire.
- **Bruit** léger, comme le bruit des muscles ou du secteur.
- **Décalage** jusqu'à 0,5 s, l'enregistrement ne commence jamais au même moment du battement.

Toutes ces modifications ne changent pas le diagnostic. Je n'ai pas mis de retournement du signal ou d'inversion des dérivations, ça changerait le sens médical du tracé.

Le générateur `g` avec une graine fixe rend l'entraînement reproductible, le même code donne le même modèle.

### La boucle d'entraînement

- **AdamW**, l'optimiseur standard, avec un `weight_decay` qui pousse les poids à rester petits (moins de surapprentissage).
- **OneCycleLR**, le taux d'apprentissage monte pendant les premiers 20% de l'entraînement puis redescend doucement. Ça converge plus vite et plus proprement qu'un taux fixe.
- Après chaque époque, je calcule la macro AUROC sur la validation et je **garde une copie des poids de la meilleure époque**. À la fin je recharge ces poids-là, pas ceux de la dernière époque.
- `predict` travaille par paquets de 256 ECG pour ne pas saturer la mémoire, et `torch.no_grad()` évite de calculer des gradients inutiles.

### Erreur 1, j'ai traité le problème comme s'il n'y avait qu'un seul diagnostic par ECG

Dans ma première version, `predict` utilisait un **softmax** et l'entraînement une **CrossEntropyLoss**. Le softmax force les 5 probabilités à faire 1 au total, comme si chaque ECG ne pouvait avoir qu'un seul diagnostic. Or un ECG peut très bien avoir un infarctus **et** une anomalie ST/T.

Le test `test_predict_gives_independent_probabilities` l'a montré (commit "Tests de l'entraînement", rouge sur GitHub). Avec un faux modèle sûr des 5 diagnostics, le softmax donnait 0,2 à chacun au lieu de probabilités proches de 1.

La correction, une **sigmoïde par diagnostic**, chacun est une question oui ou non indépendante, et la perte **BCEWithLogitsLoss** qui va avec. Elle prend directement les logits et applique la sigmoïde à l'intérieur, de façon plus stable numériquement que si je le faisais avant. C'est pour ça que le modèle de l'étape 2 renvoie des logits.

Le test sur les vraies données passait aussi avec l'erreur. Le code tournait, il donnait juste des probabilités fausses. C'est un piège important, un code qui ne plante pas n'est pas forcément un code juste.

### Erreur 2, une fuite de données dans la normalisation

Dans `run.py`, j'ai d'abord calculé la moyenne et l'écart-type de chaque dérivation sur **tout** le dataset (`ds.X`), donc aussi sur la validation et le test. Les tests passaient, aucun ne pouvait le voir. Je l'ai repéré en relisant le code.

Le problème, c'est une fuite d'information. Des statistiques calculées en partie sur les patients de test entrent dans le modèle. Ici l'effet serait petit, mais par principe tout ce que le modèle utilise doit venir du train seulement. Sinon les résultats du test ne mesurent plus vraiment ce qui se passerait sur de nouveaux patients.

### Erreur 3, j'ai choisi la meilleure époque sur le test

J'avais aussi passé le fold 10 (le test) à `fit` pour suivre l'AUROC pendant l'entraînement. Mais `fit` s'en sert pour **choisir** la meilleure époque. Dès qu'on fait un choix avec le test, le test n'est plus indépendant, et le score final serait trop optimiste.

La correction, c'est le fold 8 (la validation) qui sert à choisir. Le fold 10 n'est plus touché jusqu'à l'évaluation finale de l'étape 6.

### Ce que sauve run.py

`model.pt` contient les poids **et** la moyenne et l'écart-type du train. Les deux vont ensemble, un modèle sans sa normalisation donnerait n'importe quoi. `history.json` garde la perte et l'AUROC de validation à chaque époque, pour tracer la courbe d'entraînement plus tard.

### Les tests

- La macro AUROC vaut 1 pour un classement parfait.
- L'augmentation garde la forme du signal.
- Les probabilités sont indépendantes d'un diagnostic à l'autre.
- `fit` tourne sur le faux PTB-XL et donne une perte finie.
- La commande `python -m ecgc.run train` complète produit bien `model.pt` et `history.json`.

Le faux PTB-XL est maintenant créé une seule fois pour tous les tests, grâce à une fixture pytest dans `conftest.py`.

---

## Étape 4. Le vrai entraînement sur GitHub Actions

### Ce que j'ai ajouté, dans l'ordre des commits

1. Le workflow d'entraînement `train.yml`, qui a échoué au premier lancement
2. La correction, l'accès anonyme aux données
3. Le résumé de l'entraînement sur la page du run

### Pourquoi GitHub Actions pour entraîner

Je n'ai pas de carte graphique, et je ne voulais pas bloquer mon ordinateur pendant des heures. GitHub prête gratuitement des machines aux repos publics (4 processeurs, 16 Go de mémoire), les mêmes qui lancent déjà mes tests. Un job peut durer au maximum 6 heures, j'ai mis `timeout-minutes: 340` pour garder une marge.

`--threads $(nproc)` dit à PyTorch d'utiliser tous les processeurs de la machine. `nproc` est une commande Linux qui donne leur nombre.

### Lancé à la main, pas à chaque push

Le workflow démarre avec `workflow_dispatch`, c'est-à-dire avec le bouton "Run workflow" dans l'onglet Actions. Si je l'avais mis sur `push` comme les tests, chaque petit commit aurait relancé 2 h 30 d'entraînement. Le champ `epochs` permet de choisir le nombre d'époques au moment du lancement, sans toucher au code.

### Télécharger PTB-XL sur la machine

PhysioNet a une copie officielle de ses données sur Amazon S3 (le stockage d'AWS), dans un bucket public. C'est beaucoup plus rapide que le site. `aws s3 sync` ne copie que ce que je lui demande avec `--include`, ici les deux fichiers CSV et le dossier `records100` (les ECG à 100 Hz). Les ECG à 500 Hz ne servent à rien pour moi et pèsent bien plus lourd.

Ensuite `python -m ecgc.run prepare` fabrique le cache `.npz` de l'étape 1, et `rm -rf data/raw` supprime les fichiers bruts pour libérer de la place.

### Erreur 1, aws cherchait un compte AWS

Le premier lancement a échoué au bout de 42 secondes, avec ce message.

```
Unable to locate credentials. You can configure credentials by running "aws configure".
```

Par défaut, `aws` signe chaque demande avec un compte AWS. Sur la machine de GitHub il n'y en a pas, donc il refuse avant même d'essayer. Le bucket de PhysioNet est public, il n'y a pas besoin de compte. L'option `--no-sign-request` dit justement à `aws` de faire une demande anonyme. J'ai pu reproduire l'erreur sur mon ordinateur avant de corriger, pour être sûr que c'était bien ça.

### D'abord un test rapide avec 1 époque

Avant de lancer 20 époques, j'ai lancé le workflow avec `epochs=1`. Il a pris 15 minutes et il a vérifié toute la chaîne d'un coup, le téléchargement, la préparation du cache, l'entraînement, l'enregistrement du résultat. Si une étape avait cassé à la fin, je l'aurais su en 15 minutes et pas après 2 h 30 d'attente.

### Le cache, pour ne pas tout retélécharger

`actions/cache` garde le fichier `data/ptbxl100.npz` chez GitHub entre deux runs. Il est retrouvé grâce à sa clé, `ptbxl-1.0.3-100hz-v1`. L'étape de téléchargement a la condition `if: steps.data.outputs.cache-hit != 'true'`, elle ne tourne que si le cache n'a pas été trouvé.

Le test rapide a créé le cache. Au lancement des 20 époques, l'étape de téléchargement est marquée "skipped" et le cache a été restauré en 3 secondes. Si un jour je change le format du cache, il faudra changer la clé (`v2`), sinon GitHub redonnerait l'ancien fichier.

### Les artefacts, plutôt que commiter le modèle

À la fin, `actions/upload-artifact` enregistre le dossier `runs/main` (le modèle et `history.json`) comme artefact, téléchargeable depuis la page du run pendant 90 jours. J'aurais pu faire commiter le modèle par le workflow lui-même, mais ce commit aurait été fait par un robot (`github-actions[bot]`), qui serait apparu dans les contributeurs. Et il y aurait un fichier binaire de plus dans l'historique à chaque entraînement. Le modèle final, je l'ajouterai moi-même plus tard.

### Le résumé sur la page du run

`scripts/summary.py` lit `history.json` et écrit un tableau en markdown, une ligne par époque, la meilleure en gras. Tout ce qu'on écrit dans le fichier `$GITHUB_STEP_SUMMARY` s'affiche sur la page du run, sous la liste des jobs. Pas besoin de fouiller dans les logs.

La ligne `echo "::notice title=Entraînement::..."` crée une annotation, le petit encadré bleu en haut de la page. GitHub reconnaît les lignes qui commencent par `::notice`, `::warning` ou `::error` et les affiche à part.

### Les résultats

- 20 époques en 2 h 34 min, soit environ 7 min 40 par époque.
- Meilleure époque, la 20e, avec une macro AUROC de **0,943** sur la validation (fold 8).

La meilleure époque est la dernière. Ça ne veut pas forcément dire qu'il fallait plus d'époques. Avec `OneCycleLR`, le taux d'apprentissage devient presque nul à la fin, le modèle fait ses derniers petits ajustements et c'est souvent là qu'il est le meilleur.

Le 0,943 est un peu optimiste, parce que c'est sur la validation que j'ai choisi la meilleure époque. Le vrai score sera mesuré une seule fois sur le test (fold 10), à l'étape 6. Pour comparer, l'article de référence sur PTB-XL donne autour de 0,93 sur le test pour les meilleurs modèles, sur ces 5 diagnostics.

### Les avertissements sur la page du run

Le run est vert, mais GitHub affiche deux avertissements jaunes. Le premier dit que les versions des actions que j'utilise (`checkout@v4`, `cache@v4`...) tournent sur une ancienne version de Node.js. Le second que `ubuntu-latest` va passer à une nouvelle version d'Ubuntu. Ce ne sont pas des erreurs, rien n'a cassé, c'est GitHub qui prévient d'un changement à venir.

---

## Étape 5. La prédiction conforme

### Ce que j'ai ajouté, dans l'ordre des commits

1. Les scores de calibration
2. Les p-valeurs
3. Les réponses gardées et les quatre états
4. Les tests de la prédiction conforme, qui ont échoué
5. La correction du +1 oublié, les tests repassent au vert
6. Le contrôle du risque
7. Les tests du contrôle du risque
8. `run.py` garde les probas de calibration
9. La correction, la calibration se fait sur le fold 9

### Le problème que ça résout

Le modèle donne une proba, par exemple 0,62 pour l'infarctus. Mais 0,62 ne veut pas dire "62% de chances". Un réseau de neurones peut être trop sûr de lui ou pas assez, ses probas ne sont pas fiables telles quelles. Et surtout, il donne toujours une réponse, même quand l'ECG est ambigu.

Je voulais un modèle qui sache dire "je ne sais pas", avec une garantie chiffrée. C'est ce que fait la prédiction conforme, sans réentraîner le modèle. Elle se branche sur ses sorties.

### L'idée, comparer à des patients déjà connus

Le fold 9 sert de calibration. Ce sont des patients que le modèle n'a jamais vus, et dont je connais les vrais diagnostics. Pour chaque diagnostic, je les sépare en deux groupes, ceux qui l'ont vraiment (les positifs) et ceux qui ne l'ont pas (les négatifs).

Pour un nouveau patient, je pose deux questions.
- S'il avait le diagnostic, est-ce que sa proba ressemblerait à celles des vrais positifs ?
- S'il ne l'avait pas, est-ce qu'elle ressemblerait à celles des vrais négatifs ?

### Le score de bizarrerie et la p-valeur

Pour un vrai positif, une proba basse est bizarre, donc son score est `1 - p`. Pour un vrai négatif, une proba haute est bizarre, donc son score est `p`. `calibration_scores` calcule ces scores pour tous les patients de calibration et les trie.

La p-valeur compte combien de patients de calibration sont au moins aussi bizarres que le nouveau.

```
p-valeur = (1 + nombre de scores de calibration >= score du nouveau) / (n + 1)
```

Un exemple. J'ai 100 vrais positifs en calibration. Le nouveau patient a une proba de 0,3, donc un score de 0,7 dans l'hypothèse "positif". Si 30 vrais positifs ont un score au moins aussi grand, la p-valeur vaut (1 + 30) / 101, environ 0,31. Il ressemble assez à un positif. S'il était plus bizarre que les 100, elle vaudrait 1 / 101, presque rien.

Comme les scores sont triés, `np.searchsorted` trouve la position du nouveau score d'un coup, sans boucle sur les 100 patients.

### Les quatre états

Je garde une réponse si sa p-valeur dépasse `alpha`, par exemple 0,1.

| "présent" gardé | "absent" gardé | Réponse |
|---|---|---|
| oui | non | présent |
| non | oui | absent |
| oui | oui | je ne sais pas |
| non | non | atypique |

"Je ne sais pas", c'est quand la proba ressemble aux deux groupes, la zone grise où positifs et négatifs se mélangent. "Atypique", c'est quand elle ne ressemble à aucun des deux, une proba que presque aucun patient connu n'a eue. C'est rare, mais ça peut arriver avec un ECG très différent de ceux de PTB-XL.

### La garantie

Avec `alpha = 0,1`, un patient qui a vraiment le diagnostic garde la réponse "présent" dans au moins 90% des cas. Pareil pour les patients qui ne l'ont pas avec "absent". Ça reste vrai même si le modèle est moyen, un modèle moins bon donnera juste plus de "je ne sais pas".

Il y a une seule condition. Les patients de calibration et les nouveaux patients doivent venir du même mélange, aucun ne doit être "spécial" par rapport aux autres. On dit qu'ils sont échangeables.

"Mondrian" veut dire que la garantie est calculée séparément pour chaque diagnostic et pour chaque groupe. C'est important en santé. Une garantie de 90% en moyenne sur tout le monde pourrait cacher un modèle qui rate souvent les malades, s'ils sont peu nombreux. Ici les malades ont leur propre garantie.

### Erreur 1, j'ai oublié le +1

Ma première version de `_pvalues` renvoyait simplement `n_ge / n`, la part des patients de calibration au moins aussi bizarres. Le test `test_pvalue_formula` l'a montré (commit "Tests de la prédiction conforme", rouge sur GitHub). Sur un petit exemple calculé à la main, il attendait 0,6, 1 et 0,2, et ma fonction donnait 0,5, 1 et 0.

Le +1, c'est le nouveau patient lui-même. Avec lui, on a n + 1 patients en tout, et il est forcément au moins aussi bizarre que lui-même. Sans le +1, un patient plus bizarre que tous les autres aurait une p-valeur de 0, et la garantie n'est plus exacte. Et avec 0 patient de calibration, c'était une division par zéro.

J'ai mesuré la différence avec des simulations, 2 000 tirages avec `alpha = 0,1`. Voici la part des vrais positifs qui gardent "présent".

| Patients de calibration | Avec le +1 | Sans le +1 |
|---:|---:|---:|
| 30 | 0,950 | 0,860 |
| 100 | 0,915 | 0,885 |
| 400 | 0,904 | 0,896 |

Avec beaucoup de patients l'écart est petit, c'est pour ça que le test de couverture passait quand même avec l'erreur. Seul le test de la formule exacte l'a attrapée. Avec peu de patients, la garantie de 90% tombe à 86%. Ce que je retiens, tester une formule sur un exemple assez petit pour le calculer à la main.

### Le contrôle du risque

La deuxième méthode répond à une autre question. Je veux une liste de diagnostics à vérifier, et ne pas oublier plus de 10% des vrais diagnostics.

J'annonce tous les diagnostics dont la proba dépasse un seuil `1 - lam`. Plus `lam` est grand, plus le seuil est bas, plus j'annonce de diagnostics. J'en oublie moins, mais j'ai plus de fausses alertes. `fnr_loss` calcule, pour chaque patient, la part de ses vrais diagnostics que j'ai ratés (FNR veut dire taux de faux négatifs).

`crc_lambda` essaie 2 001 valeurs de `lam` entre 0 et 1, et garde la plus petite qui respecte cette règle sur la calibration.

```
n / (n + 1) × risque mesuré + 1 / (n + 1) <= alpha
```

Le `1 / (n + 1)` joue le même rôle que le +1 de la p-valeur, une marge pour le nouveau patient, dans le pire cas où il perd tout. La garantie, en moyenne, la part des vrais diagnostics oubliés reste sous `alpha`.

Les patients qui n'ont aucun diagnostic sont mis de côté, ils ne peuvent rien oublier.

Les deux méthodes ne servent pas à la même chose. Le conforme par label donne une réponse diagnostic par diagnostic, avec "je ne sais pas", ce sera le cœur de l'app. Le contrôle du risque donne une seule liste, pratique pour un dépistage où il ne faut surtout pas rater de malades.

### Erreur 2, j'ai calibré sur la validation

Dans `run.py`, j'ai d'abord calculé les probas de calibration sur `val`, le fold 8, parce qu'il était déjà chargé. Les tests passaient. Je l'ai vu en relisant le code.

Le problème, le fold 8 a déjà servi à choisir la meilleure époque. Le modèle gardé est celui qui marche le mieux sur ces patients-là, donc ses probas y sont un peu trop bonnes par rapport à un vrai nouveau patient. Les patients de calibration ne sont plus comme les autres, la condition de la garantie n'est plus respectée, et la couverture annoncée serait trop optimiste.

La correction, la calibration se fait sur le fold 9, qui n'a servi à rien avant. Et le test de la commande vérifie maintenant que `calibration.npz` contient exactement les patients du fold 9, pour que l'erreur ne puisse pas revenir sans que je le voie.

### Chaque fold a un seul rôle

| Folds | Rôle |
|---|---|
| 1 à 7 | train, le modèle apprend ses poids |
| 8 | validation, choisir la meilleure époque |
| 9 | calibration, la prédiction conforme |
| 10 | test, le score final, une seule fois, à l'étape 6 |

Dès qu'un fold a servi à faire un choix, il ne peut plus servir à mesurer quelque chose honnêtement.

### Les tests

- La formule de la p-valeur, sur un petit exemple calculé à la main.
- La couverture, sur 200 tirages simulés, au moins 90% en moyenne pour les positifs et pour les négatifs.
- Les quatre états.
- Plus `lam` est grand, moins j'oublie de diagnostics, et avec `lam = 1` je n'en oublie aucun.
- Le contrôle du risque, sur 200 tirages, au plus 10% de diagnostics oubliés en moyenne.
- La commande `train` sauve `calibration.npz`, avec les patients du fold 9.

Les tests de la prédiction conforme utilisent de fausses probas simulées, pas le vrai modèle. C'est voulu. La garantie ne dépend pas du modèle, donc je peux la vérifier en quelques secondes, sans rien entraîner.

### Et sur les vrais ECG ?

Pas encore. `calibration.npz` sera produit au prochain entraînement. Les vrais chiffres, la couverture sur le test et la part de "je ne sais pas", viendront à l'étape 6.
