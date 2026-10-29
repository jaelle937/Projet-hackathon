# 🎓 Formation Data Science & Machine Learning — Du zéro au projet Titanic

> Parcours complet : **NumPy → Pandas → Visualisation → SciPy → Machine Learning (scikit-learn) → Présentation de projet → Deep Learning → Projet final Titanic**
> Chaque chapitre se termine par un **exercice**. Fais-les dans l'ordre, dans un notebook Jupyter ou Google Colab.

---

## 📌 Mot du formateur (à lire avant de commencer)

**Sur les sources.** Je ne peux pas regarder les vidéos de Machine Learnia (ni d'autres) : ce cours est rédigé à partir de ma connaissance des **documentations officielles** (NumPy, pandas, Matplotlib, seaborn, SciPy, scikit-learn) et des références académiques classiques, qui sont les sources les plus fiables. Les vidéos de Machine Learnia restent un excellent complément : utilise-les **après** chaque chapitre, pour revoir un point précis, pas avant.

**Comment éviter de perdre du temps avec les vidéos.**

1. Lis le chapitre → 2. fais l'exercice → 3. seulement si tu bloques, cherche *la* vidéo sur *ce* point précis.
   Regarder passivement donne l'illusion d'apprendre. Coder, se tromper et corriger, c'est ça qui fixe les choses.

**Références fiables (à garder en favoris)**

| Sujet                                   | Source                                                                        |
| --------------------------------------- | ----------------------------------------------------------------------------- |
| NumPy                                   | https://numpy.org/doc/stable/user/absolute_beginners.html                     |
| pandas                                  | https://pandas.pydata.org/docs/user_guide/10min.html                          |
| Matplotlib                              | https://matplotlib.org/stable/users/explain/quick_start.html                  |
| seaborn                                 | https://seaborn.pydata.org/tutorial.html                                      |
| SciPy                                   | https://docs.scipy.org/doc/scipy/tutorial/index.html                          |
| scikit-learn                            | https://scikit-learn.org/stable/user_guide.html                               |
| Cours ML (Andrew Ng, Coursera/Stanford) | *Machine Learning Specialization*                                           |
| Livre                                   | *Hands-On Machine Learning* (A. Géron)                                     |
| Cours gratuit                           | Kaggle Learn (Pandas, Data Viz, Intro to ML)                                  |
| Vidéos FR                              | Machine Learnia (YouTube)                                                     |
| Vidéos EN                              | StatQuest (Josh Starmer), 3Blue1Brown (maths/réseaux de neurones), Andrew Ng |

> Les liens peuvent évoluer : en cas de lien mort, cherche le nom de la bibliothèque + "documentation".

---

## 🗺️ Feuille de route

| #  | Chapitre                              | Objectif                                                               |
| -- | ------------------------------------- | ---------------------------------------------------------------------- |
| 0  | Mise en place                         | Environnement + imports                                                |
| 1  | NumPy                                 | Calcul sur tableaux                                                    |
| 2  | Pandas                                | Manipuler des données tabulaires                                      |
| 3  | Graphiques en data science            | Savoir choisir un graphique                                            |
| 4  | Matplotlib & Seaborn                  | Les tracer                                                             |
| 5  | SciPy                                 | Calcul scientifique et statistiques                                    |
| 6  | Le Machine Learning, c'est quoi ?     | Concepts de base                                                       |
| 7  | Scikit-learn : création d'un modèle | Workflow complet                                                       |
| 8  | Les modèles supervisés              | Régression linéaire, logistique, KNN, arbres, forêts, SVM, boosting |
| 9  | Non supervisé                        | K-Means, PCA                                                           |
| 10 | Évaluation & validation croisée     | Métriques, CV, GridSearch                                             |
| 11 | Présenter un projet ML               | Structure, storytelling                                                |
| 12 | Introduction au Deep Learning         | Neurones, réseaux                                                     |
| 13 | **Projet final Titanic**        | Tout combiner                                                          |

---

# Chapitre 0 — Mise en place

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jupyter
```

Convention d'imports (universelle, à retenir par cœur) :

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

**Conseil** : Google Colab te dispense de toute installation (tout est déjà là).

---

# Chapitre 1 — NumPy

## 1.1 Pourquoi NumPy ?

NumPy fournit le **`ndarray`** : un tableau homogène (même type partout), stocké de façon compacte et manipulé en **vectorisé** (sans boucle `for`). C'est 10 à 100 fois plus rapide que des listes Python, et c'est la base de pandas, scikit-learn et du deep learning.

## 1.2 Créer des tableaux

| Fonction                                 | Rôle                                                           | Exemple                                    |
| ---------------------------------------- | --------------------------------------------------------------- | ------------------------------------------ |
| `np.array(obj, dtype=None)`            | Convertit une liste en tableau                                  | `np.array([1, 2, 3])`                    |
| `np.zeros(shape)` / `np.ones(shape)` | Tableau de 0 / de 1                                             | `np.zeros((2, 3))`                       |
| `np.full(shape, fill_value)`           | Rempli d'une valeur                                             | `np.full((2, 2), 7)`                     |
| `np.arange(start, stop, step)`         | Suite (stop exclu)                                              | `np.arange(0, 10, 2)` → `[0 2 4 6 8]` |
| `np.linspace(start, stop, num)`        | `num` points régulièrement espacés (stop **inclus**) | `np.linspace(0, 1, 5)`                   |
| `np.eye(n)`                            | Matrice identité                                               | `np.eye(3)`                              |
| `np.random.rand(d0, d1)`               | Uniforme [0, 1[                                                 | `np.random.rand(2, 3)`                   |
| `np.random.randn(d0, d1)`              | Loi normale (moy 0, écart-type 1)                              | `np.random.randn(3)`                     |
| `np.random.randint(low, high, size)`   | Entiers aléatoires                                             | `np.random.randint(0, 10, (2, 2))`       |

```python
rng = np.random.default_rng(seed=42)   # API moderne, reproductible
rng.normal(loc=0, scale=1, size=5)
```

**Paramètre clé : `dtype`** (`int64`, `float64`, `bool`…). `a.astype(float)` convertit.

## 1.3 Attributs essentiels

```python
a = np.array([[1, 2, 3], [4, 5, 6]])
a.shape   # (2, 3)   dimensions
a.ndim    # 2        nombre d'axes
a.size    # 6        nombre d'éléments
a.dtype   # int64
```

## 1.4 Indexation et découpage (slicing)

```python
a[0, 1]        # 2         ligne 0, colonne 1
a[:, 1]        # [2 5]     toute la colonne 1
a[1, :]        # [4 5 6]   toute la ligne 1
a[0:2, 1:]     # sous-tableau
a[a > 2]       # [3 4 5 6] indexation booléenne (masque) — TRÈS utilisé
```

## 1.5 Opérations vectorisées

```python
x = np.array([1, 2, 3])
x + 10      # [11 12 13]
x * x       # [1 4 9]       terme à terme
x ** 2
np.sqrt(x); np.exp(x); np.log(x)
```

### Le broadcasting

NumPy "étire" automatiquement les tableaux de formes compatibles :

```python
M = np.ones((3, 3))
v = np.array([1, 2, 3])
M + v          # v est ajouté à chaque ligne
```

Règle : on compare les dimensions **de droite à gauche** ; elles sont compatibles si égales ou si l'une vaut 1.

## 1.6 Fonctions statistiques et d'agrégation

| Fonction                                           | Paramètres importants                                                  | Exemple                  |
| -------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------ |
| `np.sum(a, axis=None)`                           | `axis=0` → par colonne, `axis=1` → par ligne                      | `a.sum(axis=0)`        |
| `np.mean`, `np.median`, `np.std`, `np.var` | `axis`, `ddof` (std : `ddof=1` pour l'écart-type d'échantillon) | `np.std(a, ddof=1)`    |
| `np.min`, `np.max`                             | `axis`                                                                | `a.max(axis=1)`        |
| `np.argmin`, `np.argmax`                       | position du min / max                                                   | `np.argmax(x)`         |
| `np.percentile(a, q)`                            | `q` entre 0 et 100                                                    | `np.percentile(x, 75)` |
| `np.cumsum(a)`                                   | somme cumulée                                                          |                          |
| `np.corrcoef(x, y)`                              | matrice de corrélation                                                 |                          |

> 🔑 **`axis`** : l'axe sur lequel on **réduit**. `axis=0` écrase les lignes (résultat : une valeur par colonne).

## 1.7 Manipuler la forme

| Fonction                                 | Rôle                                   |
| ---------------------------------------- | --------------------------------------- |
| `a.reshape(2, 3)` / `reshape(-1, 1)` | Change la forme (`-1` = "déduis-le") |
| `a.T`                                  | Transposée                             |
| `a.flatten()`                          | Aplati en 1D                            |
| `np.concatenate([a, b], axis=0)`       | Colle                                   |
| `np.vstack` / `np.hstack`            | Empile verticalement / horizontalement  |
| `np.split(a, n)`                       | Découpe                                |

⚠️ `reshape(-1, 1)` est **incontournable** en ML : scikit-learn attend `X` en 2D `(n_échantillons, n_variables)`.

## 1.8 Autres fonctions utiles

```python
np.where(x > 2, "grand", "petit")   # if/else vectorisé
np.unique(a, return_counts=True)    # valeurs uniques + effectifs
np.sort(a); np.argsort(a)
np.isnan(a); np.nan_to_num(a)       # gestion des NaN
np.dot(A, B)  # ou A @ B           # produit matriciel
np.linalg.inv(A); np.linalg.det(A)  # algèbre linéaire
```

## 1.9 Lien avec le ML

- Les **données** d'entrée `X` sont une matrice NumPy `(n_lignes, n_colonnes)`.
- La régression linéaire, c'est `y = X @ w + b` : un produit matriciel.
- Standardiser, normaliser, calculer des distances : tout se fait en vectorisé.

## ✍️ Exercice 1 — NumPy

1. Crée un tableau `notes` de 20 entiers aléatoires entre 0 et 20 (graine fixée à 0).
2. Calcule moyenne, médiane, écart-type (échantillon), min, max.
3. Affiche uniquement les notes `>= 10`, puis leur nombre.
4. Remplace par `"Admis"` / `"Refusé"` avec `np.where`.
5. Transforme `notes` en matrice `(4, 5)` et calcule la moyenne de chaque colonne puis de chaque ligne.
6. **Standardise** le tableau : `(notes - moyenne) / écart-type`. Vérifie que la nouvelle moyenne ≈ 0 et l'écart-type = 1.
7. **Bonus** : sans boucle `for`, calcule la distance euclidienne entre deux vecteurs `a = [1, 2, 3]` et `b = [4, 6, 8]`.

---

# Chapitre 2 — Pandas

## 2.1 Les deux structures

- **`Series`** : une colonne (1D, avec index).
- **`DataFrame`** : un tableau (lignes × colonnes), c'est ta structure principale.

```python
df = pd.DataFrame({
    "nom": ["Aina", "Hery", "Lalao"],
    "age": [22, 30, 27],
    "ville": ["Tana", "Tamatave", "Tana"]
})
```

## 2.2 Lire et écrire des données

| Fonction                                                                                                                                           | Paramètres clés                                                          |
| -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `pd.read_csv(path, sep=",", header=0, index_col=None, usecols=None, dtype=None, na_values=None, parse_dates=None, encoding="utf-8", nrows=None)` | `sep=";"` pour les CSV français, `encoding="latin-1"` en cas d'erreur |
| `pd.read_excel(path, sheet_name=0)`                                                                                                              |                                                                            |
| `pd.read_sql(query, connexion)`                                                                                                                  | 💡 Très utile pour toi (Oracle / SQL)                                     |
| `pd.read_json(path)`                                                                                                                             |                                                                            |
| `df.to_csv(path, index=False)`                                                                                                                   | `index=False` évite une colonne d'index inutile                         |
| `df.to_excel(path, index=False)`                                                                                                                 |                                                                            |

## 2.3 Explorer un DataFrame (le premier réflexe)

```python
df.head(n)         # n premières lignes (défaut 5)
df.tail(n)
df.shape           # (lignes, colonnes)
df.info()          # types, valeurs non nulles, mémoire
df.describe()      # stats des colonnes numériques (include="all" pour tout)
df.columns; df.dtypes; df.index
df["col"].value_counts(normalize=False, dropna=True)
df["col"].unique(); df["col"].nunique()
df.isna().sum()    # nb de valeurs manquantes par colonne
df.duplicated().sum()
```

## 2.4 Sélectionner

```python
df["age"]                      # Series
df[["nom", "age"]]             # DataFrame
df.loc[0, "nom"]               # par ÉTIQUETTES
df.loc[0:1, ["nom", "age"]]    # ⚠️ loc inclut la borne de fin
df.iloc[0, 0]                  # par POSITIONS
df.iloc[0:2, 0:2]              # iloc exclut la borne de fin
```

### Filtrer

```python
df[df["age"] > 25]
df[(df["age"] > 25) & (df["ville"] == "Tana")]    # & | ~ avec parenthèses !
df[df["ville"].isin(["Tana", "Majunga"])]
df.query("age > 25 and ville == 'Tana'")
df[df["nom"].str.contains("a", case=False)]
```

## 2.5 Créer / modifier / supprimer des colonnes

```python
df["age_dans_5_ans"] = df["age"] + 5
df["majeur"] = np.where(df["age"] >= 18, "oui", "non")
df.rename(columns={"nom": "prenom"}, inplace=False)
df.drop(columns=["majeur"])               # axis=1 équivalent
df.drop(index=[0, 1])                     # supprimer des lignes
df["age"] = df["age"].astype(float)
df["age"].apply(lambda x: x * 2)          # appliquer une fonction
df["ville"].map({"Tana": "Antananarivo"}) # remplacer via dictionnaire
df["ville"].replace("Tana", "TNR")
```

## 2.6 Valeurs manquantes (étape critique en ML)

| Fonction                                                   | Rôle                           | Paramètres                                              |
| ---------------------------------------------------------- | ------------------------------- | -------------------------------------------------------- |
| `df.isna()` / `df.notna()`                             | Masque booléen                 |                                                          |
| `df.dropna(axis=0, how="any", subset=None, thresh=None)` | Supprime                        | `subset=["age"]` : uniquement si NaN dans ces colonnes |
| `df.fillna(value, method=None)`                          | Remplace                        | `df["age"].fillna(df["age"].median())`                 |
| `df.interpolate()`                                       | Interpole (séries temporelles) |                                                          |

**Règle pratique** : médiane pour le numérique (robuste aux valeurs extrêmes), mode pour le catégoriel.

```python
df["ville"] = df["ville"].fillna(df["ville"].mode()[0])
```

## 2.7 Trier

```python
df.sort_values(by="age", ascending=False)
df.sort_values(by=["ville", "age"], ascending=[True, False])
df.sort_index()
df.nlargest(3, "age"); df.nsmallest(3, "age")
```

## 2.8 Grouper et agréger (le cœur de l'analyse)

```python
df.groupby("ville")["age"].mean()
df.groupby("ville").agg(
    age_moyen=("age", "mean"),
    effectif=("nom", "count"),
    age_max=("age", "max")
)
df.groupby(["ville", "sexe"])["age"].mean().reset_index()
```

Équivalent SQL : `SELECT ville, AVG(age) FROM t GROUP BY ville` — tu connais déjà la logique grâce à ton stage.

### Tableaux croisés

```python
pd.crosstab(df["ville"], df["sexe"], normalize="index")     # proportions par ligne
pd.pivot_table(df, values="age", index="ville", columns="sexe", aggfunc="mean")
```

## 2.9 Combiner des tableaux

| Fonction                                             | Équivalent SQL        | Paramètres                                                                                        |
| ---------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------- |
| `pd.merge(left, right, on="cle", how="inner")`     | `JOIN`               | `how` : `inner`, `left`, `right`, `outer` ; `left_on`/`right_on` si noms différents |
| `pd.concat([df1, df2], axis=0, ignore_index=True)` | `UNION ALL` (axis=0) | `axis=1` pour coller côte à côte                                                              |
| `df1.join(df2)`                                    | JOIN sur l'index       |                                                                                                    |

```python
pd.merge(commandes, clients, on="client_id", how="left")
```

## 2.10 Variables catégorielles → numériques (indispensable pour le ML)

```python
pd.get_dummies(df, columns=["ville"], drop_first=True)   # one-hot encoding
df["sexe"].map({"homme": 0, "femme": 1})                 # encodage binaire
pd.cut(df["age"], bins=[0, 18, 40, 100], labels=["jeune", "adulte", "senior"])  # découpe en classes
pd.qcut(df["age"], q=4)                                  # quartiles
```

## 2.11 Dates

```python
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
df["annee"] = df["date"].dt.year
df["mois"] = df["date"].dt.month
df["jour_semaine"] = df["date"].dt.day_name()
df.set_index("date").resample("M")["ventes"].sum()
```

## 2.12 Texte

```python
df["nom"].str.upper(); .str.strip(); .str.split(" ", expand=True)
df["nom"].str.extract(r"(\w+),")        # regex
```

## 2.13 Nettoyage — checklist type

1. `info()`, `describe()`, `isna().sum()`
2. Corriger les types (`astype`, `to_datetime`)
3. Gérer doublons : `df.drop_duplicates()`
4. Gérer NaN
5. Repérer les valeurs aberrantes (boxplot, méthode IQR)
6. Encoder les catégories

```python
Q1, Q3 = df["age"].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_sans_outliers = df[df["age"].between(Q1 - 1.5*IQR, Q3 + 1.5*IQR)]
```

## ✍️ Exercice 2 — Pandas

Crée (ou télécharge) un petit fichier `ventes.csv` avec : `date`, `produit`, `categorie`, `quantite`, `prix_unitaire`, `ville`. (Tu peux le générer avec NumPy.) Puis :

1. Charge-le, affiche `info()` et `describe()`.
2. Ajoute une colonne `chiffre_affaires = quantite * prix_unitaire`.
3. Quel est le chiffre d'affaires total par ville ? par catégorie ?
4. Quels sont les 3 produits les plus vendus (en quantité) ?
5. Convertis `date` en datetime, puis calcule le CA **par mois**.
6. Introduis volontairement 5 NaN dans `quantite` puis remplace-les par la médiane.
7. Crée un tableau croisé ville × catégorie (somme du CA).
8. Encode `categorie` en one-hot avec `get_dummies`.
9. **Bonus** : reproduis la question 3 en SQL (écris la requête) et compare avec ton résultat pandas.

---

# Chapitre 3 — Les graphiques en data science

## 3.1 Pourquoi visualiser ?

Un graphique sert à **comprendre** (exploration), **vérifier** (qualité des données, hypothèses) et **convaincre** (présentation). L'exemple classique : le *quartet d'Anscombe*, quatre jeux de données avec les mêmes statistiques (moyenne, variance, corrélation) mais des formes totalement différentes. Les chiffres seuls ne suffisent jamais.

## 3.2 Quel graphique pour quelle question ?

| Ce que tu veux savoir                     | Variables                        | Graphique                                                   |
| ----------------------------------------- | -------------------------------- | ----------------------------------------------------------- |
| Distribution d'une variable numérique    | 1 numérique                     | **Histogramme**, KDE (densité), boxplot              |
| Comparer des catégories                  | 1 catégorielle (+ 1 numérique) | **Barres** (`barplot`, `countplot`)               |
| Dispersion / valeurs extrêmes par groupe | 1 num + 1 cat                    | **Boxplot**, violin plot                              |
| Relation entre deux variables numériques | 2 numériques                    | **Nuage de points** (scatter)                         |
| Évolution dans le temps                  | temps + 1 numérique             | **Courbe** (line plot)                                |
| Corrélations entre toutes les variables  | plusieurs numériques            | **Heatmap** de corrélation                           |
| Vue d'ensemble de plusieurs variables     | plusieurs                        | **Pairplot**                                          |
| Proportions d'un tout                     | 1 cat                            | Camembert (à éviter si > 4 parts → préfère les barres) |

## 3.3 Lire les graphiques clés

- **Histogramme** : forme (symétrique, asymétrique à droite/gauche, bimodale), centre, étendue. Le nombre de classes (*bins*) change l'interprétation : teste plusieurs valeurs.
- **Boxplot** : la boîte va de Q1 à Q3 ; la ligne = médiane ; les moustaches ≈ 1,5 × IQR ; les points au-delà = valeurs aberrantes potentielles.
- **Scatter** : tendance (positive, négative, nulle), forme (linéaire, courbe), groupes, outliers.
- **Heatmap de corrélation** : valeurs entre -1 et +1. Proche de ±1 = lien linéaire fort. ⚠️ **Corrélation ≠ causalité**.

## 3.4 Bonnes pratiques

1. Un titre explicite, des axes **nommés avec unités**.
2. Choisir le bon type de graphique (voir tableau).
3. Éviter les couleurs inutiles, penser aux daltoniens.
4. Axe des barres : **toujours commencer à 0**.
5. Pas de 3D décorative.
6. Un message par graphique.

---

# Chapitre 4 — Matplotlib & Seaborn

## 4.1 Matplotlib : l'anatomie

- **Figure** : la fenêtre entière.
- **Axes** : un graphique dans la figure (ne pas confondre avec "axis").
- Deux styles : *pyplot* (rapide) et **orienté objet** (recommandé, plus propre).

```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot([1, 2, 3], [2, 4, 9], label="courbe", color="tab:blue", linestyle="--", marker="o")
ax.set_title("Mon titre")
ax.set_xlabel("x"); ax.set_ylabel("y")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

## 4.2 Les fonctions principales de Matplotlib

| Fonction                            | Usage                | Paramètres clés                                                                     |
| ----------------------------------- | -------------------- | ------------------------------------------------------------------------------------- |
| `ax.plot(x, y)`                   | Courbe               | `color`, `linestyle` (`-`, `--`, `:`), `marker`, `linewidth`, `label` |
| `ax.scatter(x, y)`                | Nuage de points      | `s` (taille), `c` (couleur / valeurs), `alpha` (transparence), `cmap`         |
| `ax.bar(x, height)` / `ax.barh` | Barres               | `width`, `color`, `yerr`                                                        |
| `ax.hist(data)`                   | Histogramme          | `bins`, `density`, `alpha`, `edgecolor`                                       |
| `ax.boxplot(data)`                | Boîte à moustaches | `labels`, `vert`                                                                  |
| `ax.pie(values)`                  | Camembert            | `labels`, `autopct="%1.1f%%"`                                                     |
| `ax.imshow(matrix)`               | Matrice / image      | `cmap`                                                                              |
| `ax.axhline(y)` / `axvline(x)`  | Ligne de référence |                                                                                       |
| `ax.annotate(...)`                | Texte avec flèche   |                                                                                       |

### Plusieurs graphiques d'un coup

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
axes[0].hist(df["age"], bins=20)
axes[1].boxplot(df["age"].dropna())
fig.suptitle("Âge des passagers")
```

Sauvegarder : `fig.savefig("graphe.png", dpi=300, bbox_inches="tight")`.

## 4.3 Seaborn : plus simple, plus joli, pensé pour pandas

Seaborn s'appuie sur Matplotlib et prend directement un DataFrame (`data=df`, `x="col"`).

```python
sns.set_theme(style="whitegrid", palette="deep")
```

### Distributions

| Fonction                                                          | Rôle                  | Paramètres                   |
| ----------------------------------------------------------------- | ---------------------- | ----------------------------- |
| `sns.histplot(data=df, x="age", bins=30, kde=True, hue="sexe")` | Histogramme + densité | `hue` colore par catégorie |
| `sns.kdeplot(data=df, x="age", hue="sexe", fill=True)`          | Densité               |                               |
| `sns.boxplot(data=df, x="classe", y="age")`                     | Boîtes                | `hue`                       |
| `sns.violinplot(...)`                                           | Boîte + densité      |                               |

### Catégories

| Fonction                                             | Rôle                                                             |
| ---------------------------------------------------- | ----------------------------------------------------------------- |
| `sns.countplot(data=df, x="classe", hue="survie")` | Effectifs par catégorie                                          |
| `sns.barplot(data=df, x="classe", y="survie")`     | Moyenne par catégorie + intervalle de confiance (barre d'erreur) |

### Relations

| Fonction                                                                                      | Rôle                          |
| --------------------------------------------------------------------------------------------- | ------------------------------ |
| `sns.scatterplot(data=df, x="age", y="prix", hue="classe", size="taille")`                  | Nuage de points                |
| `sns.lineplot(data=df, x="date", y="ventes")`                                               | Courbe                         |
| `sns.regplot(data=df, x="x", y="y")`                                                        | Nuage + droite de régression  |
| `sns.pairplot(df, hue="cible")`                                                             | Toutes les paires de variables |
| `sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", center=0, fmt=".2f")` | Matrice de corrélation        |

### Grilles de graphiques

```python
sns.catplot(data=df, x="classe", y="age", col="sexe", kind="box")
sns.relplot(data=df, x="age", y="prix", col="classe", kind="scatter")
```

(`FacetGrid`, `catplot`, `relplot` : des graphiques "en petits multiples" par catégorie.)

## 4.4 Matplotlib ou Seaborn ?

- **Seaborn** pour explorer vite et faire des graphiques statistiques propres.
- **Matplotlib** pour personnaliser finement (annotations, mise en page). On combine les deux : un graphique Seaborn retourne un `ax` Matplotlib qu'on peut encore modifier.

```python
ax = sns.histplot(df["age"], kde=True)
ax.set_title("Distribution de l'âge")
ax.axvline(df["age"].median(), color="red", linestyle="--")
```

## ✍️ Exercice 3 — Graphiques (avec le DataFrame `ventes` du chapitre 2)

1. Histogramme du `chiffre_affaires` avec `kde=True`. Que remarques-tu sur la forme ?
2. Boxplot du CA par catégorie. Y a-t-il des valeurs aberrantes ?
3. Courbe du CA mensuel (Matplotlib orienté objet, avec titre, axes et grille).
4. Barplot du CA total par ville, trié du plus grand au plus petit.
5. Heatmap de corrélation entre `quantite`, `prix_unitaire` et `chiffre_affaires`. Interprète.
6. Une figure à 2 sous-graphiques côte à côte (histogramme + boxplot de la même variable).
7. Sauvegarde ta figure en PNG à 300 dpi.
8. **Bonus** : pour chaque graphique, écris en une phrase le **message** qu'il transmet à un décideur.

---

# Chapitre 5 — SciPy et calcul scientifique

## 5.1 Rôle de SciPy

NumPy = tableaux et opérations de base. **SciPy** = boîte à outils scientifique construite dessus : statistiques, optimisation, algèbre linéaire avancée, interpolation, traitement du signal, intégration…

Principaux sous-modules :

| Module                | Contenu                                                 |
| --------------------- | ------------------------------------------------------- |
| `scipy.stats`       | Lois de probabilité, tests statistiques, corrélations |
| `scipy.optimize`    | Minimisation, ajustement de courbes                     |
| `scipy.linalg`      | Algèbre linéaire                                      |
| `scipy.integrate`   | Intégrales, équations différentielles                |
| `scipy.interpolate` | Interpolation                                           |
| `scipy.spatial`     | Distances (`distance.euclidean`, `cdist`), KD-tree  |
| `scipy.sparse`      | Matrices creuses (utile en texte/NLP)                   |
| `scipy.signal`      | Filtres, signaux                                        |

## 5.2 `scipy.stats` — le plus important pour toi

### Lois de probabilité

```python
from scipy import stats
norm = stats.norm(loc=0, scale=1)     # loi normale
norm.pdf(0)         # densité en 0
norm.cdf(1.96)      # P(X ≤ 1.96) ≈ 0.975
norm.ppf(0.975)     # quantile : ≈ 1.96
norm.rvs(size=5, random_state=0)   # tirages aléatoires
```

Méthodes communes : `pdf`/`pmf`, `cdf`, `ppf`, `rvs`, `mean`, `std`.

### Statistiques descriptives

```python
stats.describe(x)
stats.zscore(x)            # standardisation
stats.skew(x); stats.kurtosis(x)
```

### Corrélations

```python
stats.pearsonr(x, y)       # linéaire → (r, p-value)
stats.spearmanr(x, y)      # rangs, relation monotone, robuste aux outliers
```

### Tests d'hypothèses (idée générale)

On part de **H₀** (« il n'y a pas d'effet / pas de différence »). La **p-value** est la probabilité d'observer un résultat au moins aussi extrême **si H₀ est vraie**. Si p < α (souvent 0,05), on rejette H₀.
⚠️ Une p-value faible ne mesure ni la taille de l'effet ni son importance pratique.

| Question                                         | Test                        | Fonction                                   |
| ------------------------------------------------ | --------------------------- | ------------------------------------------ |
| Moyenne d'un groupe ≠ valeur donnée ?          | t de Student 1 échantillon | `stats.ttest_1samp(x, popmean)`          |
| Deux groupes ont-ils des moyennes différentes ? | t de Student                | `stats.ttest_ind(a, b, equal_var=False)` |
| Mesures avant/après sur les mêmes individus    | t apparié                  | `stats.ttest_rel(a, b)`                  |
| Deux variables catégorielles indépendantes ?   | χ²                        | `stats.chi2_contingency(table)`          |
| Plus de 2 groupes (moyennes)                     | ANOVA                       | `stats.f_oneway(a, b, c)`                |
| La variable suit-elle une loi normale ?          | Shapiro-Wilk                | `stats.shapiro(x)`                       |
| Alternative non paramétrique au t-test          | Mann-Whitney                | `stats.mannwhitneyu(a, b)`               |

Exemple :

```python
stat, p = stats.ttest_ind(df[df.sexe=="F"]["age"], df[df.sexe=="H"]["age"], equal_var=False)
print(f"t={stat:.2f}, p={p:.4f}")
```

## 5.3 `scipy.optimize`

```python
from scipy.optimize import minimize, curve_fit

# Minimiser f(x) = (x-3)^2
res = minimize(lambda x: (x[0] - 3)**2, x0=[0.0])
res.x       # ≈ [3.]

# Ajuster une courbe y = a * exp(b * x)
def modele(x, a, b): return a * np.exp(b * x)
params, cov = curve_fit(modele, x_data, y_data)
```

💡 Idée à retenir : **entraîner un modèle de ML = minimiser une fonction de coût**. `minimize` est la version « manuelle » de ce que font les algorithmes d'apprentissage.

## 5.4 Autres exemples rapides

```python
from scipy.integrate import quad
quad(lambda x: x**2, 0, 1)            # (0.333..., erreur)

from scipy.linalg import solve
solve(A, b)                           # résout A x = b

from scipy.spatial.distance import cdist
cdist(A, B, metric="euclidean")       # matrice des distances
```

## ✍️ Exercice 4 — SciPy

1. Génère 200 valeurs d'une loi normale (moyenne 50, écart-type 10). Teste la normalité avec `shapiro`.
2. Calcule la probabilité qu'une valeur soit supérieure à 65 (avec `norm.cdf`).
3. Compare avec la proportion observée dans ton échantillon.
4. Crée deux groupes A et B (moyennes 50 et 55). Réalise un `ttest_ind`. Conclus avec α = 0,05.
5. Calcule le Pearson et le Spearman entre deux variables liées de façon non linéaire (par ex. `y = x**3 + bruit`). Lequel est le plus élevé ? Pourquoi ?
6. Avec `chi2_contingency`, teste si `sexe` et `survie` (Titanic) sont indépendants.
7. **Bonus** : retrouve la pente et l'ordonnée d'une droite avec `curve_fit`.

---

# Chapitre 6 — Le Machine Learning, c'est quoi ?

## 6.1 Définition

> Le Machine Learning (ML) est un ensemble de méthodes où un programme **apprend des règles à partir de données** au lieu d'être programmé explicitement avec ces règles.

Programmation classique : `Règles + Données → Résultats`
Machine Learning : `Données + Résultats → Règles (= le modèle)`

Le ML est un sous-domaine de l'**IA**. Le **Deep Learning** est un sous-domaine du ML (réseaux de neurones profonds).

## 6.2 Vocabulaire indispensable

| Terme                                              | Signification                                                               |
| -------------------------------------------------- | --------------------------------------------------------------------------- |
| **Observation / échantillon**               | Une ligne du dataset                                                        |
| **Features (variables explicatives), `X`** | Les colonnes d'entrée                                                      |
| **Target (cible), `y`**                    | Ce qu'on veut prédire                                                      |
| **Modèle**                                  | La fonction apprise `y ≈ f(X)`                                           |
| **Entraînement (fit)**                      | Phase où le modèle apprend                                                |
| **Prédiction (predict)**                    | Utilisation du modèle sur de nouvelles données                            |
| **Paramètres**                              | Appris par le modèle (ex. coefficients)                                    |
| **Hyperparamètres**                         | Choisis **par toi** avant l'entraînement (ex. profondeur d'un arbre) |
| **Fonction de coût (loss)**                 | Mesure l'erreur que le modèle cherche à minimiser                         |
| **Généralisation**                         | Capacité à bien prédire sur des données **jamais vues**           |

## 6.3 Les grands types d'apprentissage

1. **Supervisé** : on a la cible `y`.
   - **Régression** : `y` numérique continue (prix, température, salaire).
   - **Classification** : `y` catégorielle (spam/non-spam, survivant/non, maladie A/B/C).
2. **Non supervisé** : pas de cible ; on cherche une structure.
   - **Clustering** (regrouper : segmentation clients).
   - **Réduction de dimension** (PCA).
   - **Détection d'anomalies** (fraude).
3. **Par renforcement** : un agent apprend par essai/erreur avec des récompenses (jeux, robotique). Hors du périmètre de ce cours.

## 6.4 Le problème n°1 : surapprentissage vs sous-apprentissage

- **Sous-apprentissage (underfitting / biais élevé)** : modèle trop simple, mauvais partout.
- **Surapprentissage (overfitting / variance élevée)** : le modèle « apprend par cœur » le train (très bon score) mais échoue sur de nouvelles données.
- **Objectif** : le bon compromis (**compromis biais-variance**).

> 🔑 **Règle d'or** : on ne juge **jamais** un modèle sur les données qui ont servi à l'entraîner.

Pour cela : séparer **train** (apprendre) et **test** (évaluer, une seule fois à la fin). Et en pratique ajouter une **validation** (réglages) ou une **validation croisée**.

## 6.5 Les étapes d'un projet ML

1. Comprendre le **problème métier** et la métrique de succès.
2. Collecter / charger les données.
3. **EDA** (analyse exploratoire) : stats, graphiques, valeurs manquantes.
4. Préparer les données (nettoyage, encodage, mise à l'échelle).
5. Découper train/test.
6. Entraîner un **modèle de référence (baseline)** simple.
7. Entraîner et comparer d'autres modèles.
8. Évaluer (CV, métriques adaptées), optimiser les hyperparamètres.
9. Évaluer **une fois** sur le test.
10. Interpréter, présenter, déployer / surveiller.

---

# Chapitre 7 — Scikit-learn : créer un modèle de A à Z

## 7.1 L'API unifiée de scikit-learn

Tous les objets suivent la même logique, c'est ce qui rend la bibliothèque agréable :

| Méthode                              | Rôle                                                            |
| ------------------------------------- | ---------------------------------------------------------------- |
| `.fit(X, y)`                        | Apprend (modèles) ou calcule des statistiques (transformateurs) |
| `.predict(X)`                       | Prédit (modèles)                                               |
| `.transform(X)`                     | Transforme (transformateurs)                                     |
| `.fit_transform(X)`                 | `fit` + `transform`                                          |
| `.score(X, y)`                      | Score par défaut                                                |
| `.predict_proba(X)`                 | Probabilités (classification)                                   |
| `.get_params()` / `.set_params()` | Hyperparamètres                                                 |

## 7.2 Séparer train / test

```python
from sklearn.model_selection import train_test_split

X = df.drop(columns="cible")
y = df["cible"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20 % pour le test
    random_state=42,      # reproductibilité
    stratify=y            # garde les proportions de classes (classification)
)
```

## 7.3 Prétraitement

### Imputation (valeurs manquantes)

```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy="median")    # "mean", "most_frequent", "constant"
```

### Mise à l'échelle

| Classe               | Formule                 | Usage                                           |
| -------------------- | ----------------------- | ----------------------------------------------- |
| `StandardScaler()` | `(x − μ) / σ`      | Cas général (régression, SVM, KNN, réseaux) |
| `MinMaxScaler()`   | ramène dans [0, 1]     | Quand on veut des bornes                        |
| `RobustScaler()`   | utilise médiane et IQR | Beaucoup d'outliers                             |

Les **arbres et forêts** n'en ont pas besoin ; les modèles basés sur les **distances ou le gradient** (KNN, SVM, régression régularisée, réseaux) **oui**.

### Encodage des catégories

```python
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder
OneHotEncoder(handle_unknown="ignore", drop="first")
```

### ⚠️ Fuite de données (data leakage)

Toujours `fit` le scaler/imputer **sur le train uniquement**, puis `transform` sur le test. Sinon des informations du test « fuient » dans l'entraînement et ton score est faussement optimiste.

```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)   # fit sur train
X_test_s  = scaler.transform(X_test)        # transform seulement
```

Mieux : utiliser les **Pipelines** qui font cela automatiquement.

## 7.4 Pipelines et ColumnTransformer

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

num_cols = ["age", "prix"]
cat_cols = ["sexe", "classe"]

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])

model = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000))])
model.fit(X_train, y_train)
model.predict(X_test)
```

Avantages : pas de fuite, code propre, un seul objet à sauvegarder (`joblib.dump(model, "model.pkl")`).

## 7.5 Le workflow minimal en 5 lignes

```python
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
```

---

# Chapitre 8 — Les modèles supervisés

## 8.1 Régression linéaire (`LinearRegression`)

**Idée** : trouver la droite (ou l'hyperplan) `ŷ = w₁x₁ + … + wₙxₙ + b` qui minimise la somme des carrés des erreurs (moindres carrés).

```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()          # fit_intercept=True par défaut
lr.fit(X_train, y_train)
lr.coef_, lr.intercept_          # w et b
```

- ✅ Simple, interprétable, rapide.
- ❌ Suppose une relation linéaire ; sensible aux outliers et à la multicolinéarité.
- **Régularisation** pour limiter le surapprentissage :
  - `Ridge(alpha=1.0)` : pénalité L2, réduit les coefficients.
  - `Lasso(alpha=1.0)` : pénalité L1, peut mettre des coefficients à 0 (sélection de variables).
  - `ElasticNet(alpha, l1_ratio)` : mélange.
  - Plus `alpha` est grand, plus le modèle est simple.

## 8.2 Régression logistique (`LogisticRegression`)

Malgré son nom, c'est un modèle de **classification**. Il estime la **probabilité** d'appartenir à une classe via la fonction sigmoïde `p = 1 / (1 + e^{-(wx+b)})`.

```python
LogisticRegression(C=1.0, penalty="l2", max_iter=1000, class_weight=None)
```

- `C` : **inverse** de la régularisation (petit C = plus de régularisation).
- `class_weight="balanced"` : utile si classes déséquilibrées.
- `predict_proba(X)` : probabilités ; seuil par défaut 0,5 (modifiable selon le besoin métier).
- ✅ Très bon modèle de **baseline**, interprétable (odds ratios).

## 8.3 K plus proches voisins (`KNeighborsClassifier` / `Regressor`)

**Idée** : pour prédire un point, regarder ses `k` voisins les plus proches et voter (classification) ou faire la moyenne (régression).

```python
KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski")
```

- `k` petit → surapprentissage ; `k` grand → sous-apprentissage.
- 🔑 **Standardiser** impérativement (sinon la variable à grande échelle domine).
- ❌ Lent sur gros jeux de données (calcule toutes les distances).

## 8.4 Arbres de décision (`DecisionTreeClassifier` / `Regressor`)

**Idée** : suite de questions « si-alors » qui découpent les données (ex. « sexe = femme ? classe ≤ 2 ? »). Chaque découpage maximise la « pureté » (critère `gini` ou `entropy`).

```python
DecisionTreeClassifier(max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion="gini")
```

- Hyperparamètres anti-surapprentissage : **`max_depth`**, `min_samples_leaf`.
- ✅ Visualisable (`plot_tree`), pas besoin de normaliser, gère le non-linéaire.
- ❌ Instable et très sujet au surapprentissage.

## 8.5 Forêts aléatoires (`RandomForestClassifier` / `Regressor`)

**Idée** : **beaucoup d'arbres** entraînés sur des échantillons aléatoires (*bagging*) avec des sous-ensembles de variables ; on moyenne / vote. La diversité réduit la variance.

```python
RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=1,
                       max_features="sqrt", random_state=42, n_jobs=-1)
```

- `rf.feature_importances_` : importance des variables.
- ✅ Excellent modèle par défaut, robuste, peu de réglages.
- ❌ Moins interprétable, plus lourd.

## 8.6 Gradient Boosting (`GradientBoostingClassifier`, `HistGradientBoostingClassifier`)

**Idée** : construire des arbres **l'un après l'autre**, chacun corrigeant les erreurs du précédent.

```python
HistGradientBoostingClassifier(learning_rate=0.1, max_iter=100, max_depth=None)
```

- `learning_rate` faible + plus d'itérations = souvent mieux mais plus lent.
- Les bibliothèques **XGBoost**, **LightGBM**, **CatBoost** sont les versions les plus utilisées en compétition (Kaggle, hackathons). À connaître pour la suite.

## 8.7 SVM (`SVC` / `SVR`)

**Idée** : trouver la frontière qui **maximise la marge** entre les classes. Avec un *noyau* (kernel), peut séparer des données non linéaires.

```python
SVC(C=1.0, kernel="rbf", gamma="scale", probability=False)
```

- `C` : tolérance aux erreurs (grand C = frontière stricte, risque de surapprentissage).
- `kernel` : `linear`, `rbf`, `poly`.
- Standardisation indispensable. Lent sur très gros volumes.

## 8.8 Naïve Bayes (`GaussianNB`, `MultinomialNB`)

Basé sur le théorème de Bayes avec l'hypothèse (« naïve ») d'indépendance entre variables. Très rapide, souvent utilisé pour le **texte** (spam).

## 8.9 Tableau de synthèse

| Modèle                | Type           | Standardiser ?        | Interprétable | Point fort              | Point faible                 |
| ---------------------- | -------------- | --------------------- | -------------- | ----------------------- | ---------------------------- |
| Régression linéaire  | Régression    | Oui (si régularisé) | ⭐⭐⭐         | Simple, rapide          | Linéaire seulement          |
| Régression logistique | Classification | Oui                   | ⭐⭐⭐         | Baseline, probabilités | Frontière linéaire         |
| KNN                    | Les deux       | **Oui**         | ⭐⭐           | Intuitif                | Lent, sensible à l'échelle |
| Arbre                  | Les deux       | Non                   | ⭐⭐⭐         | Visuel                  | Surapprend                   |
| Forêt aléatoire      | Les deux       | Non                   | ⭐⭐           | Robuste                 | Moins explicable             |
| Gradient boosting      | Les deux       | Non                   | ⭐             | Très performant        | Réglages, risque d'overfit  |
| SVM                    | Les deux       | **Oui**         | ⭐             | Bon en haute dimension  | Lent sur gros volumes        |
| Naïve Bayes           | Classification | Non                   | ⭐⭐           | Rapide, texte           | Hypothèse d'indépendance   |

> 💡 **Stratégie pro** : toujours commencer par une **baseline** très simple (régression logistique / linéaire, ou même "prédire la classe majoritaire" via `DummyClassifier`). Un modèle complexe n'a de valeur que s'il bat clairement la baseline.

---

# Chapitre 9 — Apprentissage non supervisé

## 9.1 K-Means (clustering)

Regroupe les observations en `k` groupes en minimisant la distance aux centres (centroïdes).

```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X_scaled)
km.cluster_centers_, km.inertia_
```

- **Choisir `k`** : méthode du coude (inertie en fonction de `k`) et `silhouette_score` (plus proche de 1 = mieux).
- Standardiser les données avant.

## 9.2 Autres méthodes de clustering à connaître

- **`DBSCAN`** : trouve des groupes de forme libre et détecte le bruit (pas besoin de fixer `k`).
- **`AgglomerativeClustering`** : clustering hiérarchique (dendrogramme).

## 9.3 PCA — Analyse en composantes principales

Réduit le nombre de variables en conservant un maximum de variance. Utile pour visualiser en 2D et réduire le bruit.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
pca.explained_variance_ratio_      # part de variance expliquée par composante
```

## 9.4 Détection d'anomalies

`IsolationForest(contamination=0.05)` : isole les points « différents » (fraude, pannes).

---

# Chapitre 10 — Évaluation et validation

## 10.1 Métriques de régression

| Métrique                              | Lecture                                                                 | Fonction                               |
| -------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| **MAE** (erreur absolue moyenne) | Erreur moyenne en unités de `y`                                      | `mean_absolute_error`                |
| **MSE / RMSE**                   | Pénalise fortement les grosses erreurs                                 | `mean_squared_error` (RMSE = racine) |
| **R²**                          | Part de variance expliquée (1 = parfait, 0 = pas mieux que la moyenne) | `r2_score`                           |

## 10.2 Métriques de classification

### Matrice de confusion

|                          | Prédit positif   | Prédit négatif   |
| ------------------------ | ----------------- | ------------------ |
| **Réel positif**  | VP (vrai positif) | FN (faux négatif) |
| **Réel négatif** | FP (faux positif) | VN (vrai négatif) |

| Métrique                 | Formule                 | Question à laquelle elle répond                                                      |
| ------------------------- | ----------------------- | -------------------------------------------------------------------------------------- |
| **Accuracy**        | (VP+VN) / total         | Quelle proportion de bonnes prédictions ? ⚠️ trompeuse si classes déséquilibrées |
| **Précision**      | VP / (VP+FP)            | Parmi mes prédictions positives, combien sont justes ?                                |
| **Rappel (recall)** | VP / (VP+FN)            | Parmi les vrais positifs, combien ai-je trouvés ?                                     |
| **F1-score**        | 2·P·R / (P+R)         | Compromis précision/rappel                                                            |
| **ROC-AUC**         | Aire sous la courbe ROC | Capacité à classer positifs au-dessus des négatifs (0,5 = hasard, 1 = parfait)      |

**Exemple pour comprendre l'accuracy trompeuse** : si 1 % des transactions sont frauduleuses, un modèle qui répond toujours « pas de fraude » a 99 % d'accuracy… et ne sert à rien. Il faut regarder le **rappel**.

**Choisir selon le coût de l'erreur** :

- Dépistage médical, fraude → on veut un **rappel élevé** (ne pas rater).
- Filtre anti-spam, accusation → on veut une **précision élevée** (éviter les faux positifs).

```python
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay, RocCurveDisplay)

print(classification_report(y_test, y_pred))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
```

## 10.3 Validation croisée (Cross-Validation)

Un seul découpage train/test dépend du hasard. La **k-fold CV** découpe les données en `k` parts : on entraîne sur `k-1`, on teste sur la dernière, et on répète `k` fois → moyenne et écart-type des scores.

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
print(scores.mean(), scores.std())
```

- `scoring` : `"accuracy"`, `"f1"`, `"roc_auc"`, `"neg_mean_squared_error"`, `"r2"`…
- `StratifiedKFold` conserve les proportions de classes (classification).
- `cross_validate` renvoie plusieurs métriques + temps.
- `TimeSeriesSplit` pour les séries temporelles (jamais mélanger le passé et le futur).

## 10.4 Optimisation des hyperparamètres

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

param_grid = {
    "clf__n_estimators": [100, 300],
    "clf__max_depth": [None, 5, 10],
    "clf__min_samples_leaf": [1, 5],
}
gs = GridSearchCV(model, param_grid, cv=5, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)
gs.best_params_, gs.best_score_
best = gs.best_estimator_
```

- `GridSearchCV` : teste toutes les combinaisons (exhaustif).
- `RandomizedSearchCV(n_iter=...)` : tire au hasard (plus rapide si beaucoup d'hyperparamètres).
- Dans un `Pipeline`, on préfixe avec le nom de l'étape (`clf__param`).

## 10.5 Diagnostiquer son modèle

- **Courbe d'apprentissage** (`learning_curve`) : score train vs validation selon la taille des données.
  - Score train élevé, validation faible → **surapprentissage**.
  - Les deux faibles → **sous-apprentissage**.
- **Importance des variables** : `feature_importances_`, `permutation_importance`.
- **Analyse des erreurs** : regarder les cas mal prédits.

## 10.6 Procédure d'évaluation propre (à appliquer à chaque projet)

1. Séparer un **jeu de test** et ne plus y toucher.
2. Comparer plusieurs modèles par **validation croisée** sur le train.
3. Régler les hyperparamètres (GridSearch) sur le train.
4. Évaluer **une seule fois** le meilleur modèle sur le test.
5. Comparer à la **baseline**.

## ✍️ Exercice 5 — Premier modèle complet (sur un jeu de données intégré à scikit-learn)

Utilise `sklearn.datasets.load_breast_cancer(as_frame=True)` (classification) :

1. Explore : dimensions, répartition de la cible, corrélations (heatmap).
2. Split train/test stratifié (80/20).
3. Entraîne un `DummyClassifier(strategy="most_frequent")` : ta baseline. Note son accuracy.
4. Entraîne une régression logistique **dans un pipeline avec StandardScaler**.
5. Évalue en CV 5 plis (accuracy, recall, f1).
6. Compare avec `RandomForestClassifier` et `KNeighborsClassifier`.
7. Fais un `GridSearchCV` sur le meilleur modèle.
8. Affiche la matrice de confusion et le rapport de classification sur le test.
9. Dans ce contexte médical, **quelle métrique privilégier ? Pourquoi ?** Écris 3 phrases.
10. **Bonus** : refais l'exercice sur `load_diabetes(as_frame=True)` (régression) avec `Ridge`/`Lasso` et les métriques MAE, RMSE, R².

## ✍️ Exercice 6 — Non supervisé

1. Charge `load_iris`. Standardise les 4 variables, ignore la cible.
2. Applique `KMeans` pour `k` de 2 à 8 ; trace le coude et le `silhouette_score`.
3. Choisis `k`, puis compare les clusters aux vraies espèces avec `pd.crosstab`.
4. Réduis en 2D avec `PCA` et fais un scatter coloré par cluster.

---

# Chapitre 11 — Comment présenter un projet de Machine Learning

Un bon modèle mal présenté ne convainc personne. Un jury de hackathon, un manager ou un recruteur regarde d'abord le **problème résolu** et la **clarté du raisonnement**, pas la complexité de l'algorithme.

## 11.1 Structure recommandée (10 à 15 minutes)

| # | Section                                   | Contenu                                                                                  | Durée  |
| - | ----------------------------------------- | ---------------------------------------------------------------------------------------- | ------- |
| 1 | **Contexte & problème**            | Qui a le problème ? Pourquoi c'est important ? Quel impact (argent, temps, vies) ?      | 1-2 min |
| 2 | **Objectif & métrique de succès** | Ce que le modèle doit prédire et comment on mesure la réussite (en langage métier)   | 1 min   |
| 3 | **Les données**                    | Source, taille, variables, qualité, limites, biais éventuels                           | 1-2 min |
| 4 | **Exploration (EDA)**               | 2 à 3 graphiques **marquants** et leurs enseignements                             | 2 min   |
| 5 | **Préparation**                    | Nettoyage, valeurs manquantes, nouvelles variables créées (feature engineering)        | 1 min   |
| 6 | **Modélisation**                   | Baseline, modèles testés, méthode de validation                                       | 2 min   |
| 7 | **Résultats**                      | Tableau comparatif, métrique principale, matrice de confusion, importance des variables | 2 min   |
| 8 | **Interprétation & limites**       | Ce que le modèle a compris, où il se trompe, risques                                   | 1 min   |
| 9 | **Recommandations & suite**         | Actions concrètes, déploiement, amélioration                                          | 1 min   |

## 11.2 Règles de storytelling

1. **Commence par le « pourquoi »**, pas par le code.
2. **Traduis les métriques** : « Notre modèle identifie 8 clients à risque sur 10 » est plus parlant que « recall = 0,80 ».
3. **Compare toujours à une référence** (baseline, situation actuelle).
4. **Un graphique = un message** (titre-phrase : « Les femmes de 1ʳᵉ classe ont survécu à 97 % »).
5. **Sois honnête sur les limites** (données limitées, biais, risque de surapprentissage). Ça renforce ta crédibilité.
6. Termine par **une recommandation claire**, pas par « merci ».

## 11.3 Ce que les jurys valorisent

- Un problème bien cadré et une démarche **reproductible**.
- Une **baseline** et une comparaison honnête de modèles.
- Une validation correcte (pas de fuite de données, test gardé à part).
- Une **interprétation** des résultats et des limites.
- Une démo ou une visualisation parlante.

## 11.4 Modèle de README pour GitHub

```markdown
# Titre du projet
## Contexte et objectif
## Données (source, description des variables)
## Méthodologie (nettoyage, features, modèles, validation)
## Résultats (tableau + 1 ou 2 graphiques)
## Limites et pistes d'amélioration
## Comment reproduire (requirements.txt, ordre d'exécution des notebooks)
```

## 11.5 Bonnes pratiques techniques

- Fixer `random_state` partout (reproductibilité).
- Notebook structuré avec titres et commentaires courts ; code dans des fonctions.
- Sauvegarder le pipeline (`joblib.dump`).
- Environnement documenté (`requirements.txt`).
- Versionner avec Git.

## ✍️ Exercice 7 — Présentation

Rédige le **plan de présentation de 8 diapositives** (un titre-message par diapositive, pas le contenu complet) pour ton exercice sur le cancer du sein ou ton projet Titanic. Pour chaque diapositive, écris la **phrase clé** que le public doit retenir.

---

# Chapitre 12 — Petite introduction au Deep Learning

## 12.1 Définition

Le **Deep Learning** utilise des **réseaux de neurones artificiels à plusieurs couches** pour apprendre automatiquement des représentations complexes à partir des données brutes (images, texte, son).

**Quand le préférer au ML classique ?**

- Données **non structurées** (images, audio, texte) et **grands volumes**.
- Sur des **tableaux de données** (comme Titanic), les forêts aléatoires et le gradient boosting battent généralement les réseaux de neurones.

## 12.2 Le neurone artificiel

Un neurone calcule : `sortie = activation(w₁x₁ + w₂x₂ + … + b)`

- **Poids `w`** et **biais `b`** : appris.
- **Fonction d'activation** : introduit la non-linéarité.
  - `ReLU` : `max(0, x)`, standard pour les couches cachées.
  - `Sigmoïde` : sortie entre 0 et 1 (classification binaire).
  - `Softmax` : probabilités sur plusieurs classes.

## 12.3 Un réseau de neurones

```
Entrée (features) → [couches cachées] → Sortie (prédiction)
```

- **Couche dense (fully connected)** : chaque neurone est relié à tous ceux de la couche précédente.
- **Profondeur** = nombre de couches cachées ; **largeur** = neurones par couche.

## 12.4 Comment il apprend

1. **Propagation avant (forward)** : les données traversent le réseau → prédiction.
2. **Fonction de coût (loss)** : mesure l'erreur (`MSE` en régression, `cross-entropy` en classification).
3. **Rétropropagation (backpropagation)** : calcule la contribution de chaque poids à l'erreur (via la règle de dérivation en chaîne).
4. **Descente de gradient** : met à jour les poids pour réduire l'erreur : `w ← w − η · ∂Loss/∂w` (`η` = learning rate).
5. On répète sur de nombreuses **époques** (passages complets sur les données), par **mini-lots (batch)**.

Optimiseurs courants : `SGD`, `Adam` (le plus utilisé par défaut).

## 12.5 Surapprentissage et solutions

- **Dropout** (éteindre aléatoirement des neurones), **régularisation L2**, **early stopping** (arrêter quand la validation ne s'améliore plus), plus de données, augmentation de données.

## 12.6 Les grandes familles

| Architecture                           | Usage                                                 |
| -------------------------------------- | ----------------------------------------------------- |
| **MLP** (perceptron multicouche) | Données tabulaires simples                           |
| **CNN** (convolutifs)            | Images, vision par ordinateur                         |
| **RNN / LSTM**                   | Séquences (séries temporelles, texte ancien)        |
| **Transformers**                 | Texte, langage (GPT, BERT), de plus en plus la vision |
| **Autoencodeurs**                | Compression, anomalies                                |
| **GAN / modèles de diffusion**  | Génération d'images                                 |

## 12.7 Les bibliothèques

- **TensorFlow / Keras** (Google) : API haut niveau, très accessible.
- **PyTorch** (Meta) : très utilisé en recherche et de plus en plus en industrie.

## 12.8 Premier réseau avec Keras (exemple minimal)

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid"),     # classification binaire
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2, epochs=50, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
)
model.evaluate(X_test_scaled, y_test)
```

Les données doivent être **numériques et standardisées** avant d'entrer dans un réseau.

Alternative sans installer TensorFlow : `sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500)`.

## 12.9 Ressources pour aller plus loin

- 3Blue1Brown : série « Neural Networks » (intuition visuelle, en anglais, excellente).
- Andrew Ng : *Deep Learning Specialization* (Coursera).
- Documentation Keras : https://keras.io/getting_started/
- Livre : *Deep Learning with Python* (F. Chollet, créateur de Keras).

## ✍️ Exercice 8 — Deep Learning (facultatif mais recommandé)

1. Entraîne un `MLPClassifier` de scikit-learn sur le dataset du cancer du sein (données standardisées).
2. Compare ses résultats (CV) à la régression logistique et à la forêt aléatoire.
3. En 3 phrases : le réseau est-il meilleur ? Était-ce utile sur ce type de données ?

---

# 🚢 Chapitre 13 — PROJET FINAL : Les passagers du Titanic

Tu vas maintenant combiner **tout** ce que tu as appris.

## 13.1 Le contexte (mission)

> **Tu es data scientist junior dans un musée maritime international.**
> Le musée prépare une grande exposition interactive sur le naufrage du Titanic (1912). La direction veut une borne tactile : le visiteur saisit un profil (âge, sexe, classe du billet, tarif, famille à bord, port d'embarquement) et la borne affiche **la probabilité de survie** d'un passager ayant ce profil, avec une explication simple.
>
> **Ta mission :**
>
> 1. Construire un modèle de classification qui prédit la survie.
> 2. Identifier les **facteurs qui ont le plus influencé** la survie (pour le texte pédagogique de l'exposition).
> 3. Présenter ta démarche et tes résultats à la direction (non-technique) en 10 minutes.
>
> **Contraintes :** le visiteur doit pouvoir comprendre le résultat. Le musée préfère un modèle **explicable** à un modèle légèrement meilleur mais opaque. Le musée accepte une prédiction fausse sur un profil, mais veut éviter de **dire à tort** que des profils auraient « sûrement survécu » : suis donc attentivement la précision et pas seulement l'accuracy.

**Données** : le jeu de données se charge facilement avec seaborn (`sns.load_dataset("titanic")`) ou depuis Kaggle (compétition « Titanic – Machine Learning from Disaster », fichiers `train.csv` / `test.csv`).

### Dictionnaire des variables (version Kaggle)

| Variable     | Description                               |
| ------------ | ----------------------------------------- |
| `Survived` | **Cible** : 0 = mort, 1 = survivant |
| `Pclass`   | Classe du billet (1, 2, 3)                |
| `Name`     | Nom (contient le titre : Mr, Mrs, Miss…) |
| `Sex`      | Sexe                                      |
| `Age`      | Âge                                      |
| `SibSp`    | Nb de frères/sœurs/conjoint à bord     |
| `Parch`    | Nb de parents/enfants à bord             |
| `Ticket`   | Numéro de ticket                         |
| `Fare`     | Prix du billet                            |
| `Cabin`    | Numéro de cabine (beaucoup de NaN)       |
| `Embarked` | Port d'embarquement (C, Q, S)             |

## 13.2 Parcours guidé en 9 étapes

Pour chaque étape : une **mission**, des **questions à te poser** et les **outils du cours à utiliser**. À toi d'écrire le code. Quand tu as fini une étape, envoie-moi ton code ou tes résultats et je te corrige comme un relecteur.

### Étape 1 — Cadrage

- Écris en 3 phrases : le problème, la cible, la métrique principale choisie (et pourquoi).
- Type de problème : régression ou classification ? Supervisé ou non ?

### Étape 2 — Chargement et premier regard

- Outils : `read_csv`, `shape`, `info()`, `describe(include="all")`, `isna().sum()`.
- Questions : combien de lignes ? Quelle part de survivants ? Quelles colonnes ont des valeurs manquantes, et combien (en %) ? Les classes sont-elles déséquilibrées ?

### Étape 3 — Analyse exploratoire (EDA)

- Outils : `groupby`, `crosstab`, `countplot`, `histplot`, `boxplot`, `heatmap`.
- Questions à explorer :
  - Taux de survie selon `Sex`, `Pclass`, `Embarked`.
  - Taux de survie combiné `Sex × Pclass`.
  - Distribution de l'âge des survivants vs non-survivants.
  - Le prix du billet est-il lié à la classe ? À la survie ?
  - Les corrélations entre variables numériques.
- Livrable : **3 graphiques** avec un titre-message chacun.
- Test statistique (SciPy) : `chi2_contingency` entre `Sex` et `Survived`.

### Étape 4 — Nettoyage

- Outils : `fillna`, `SimpleImputer`, `drop`.
- Questions : comment traiter `Age` (quelle stratégie ? par la médiane globale ou par groupe) ? `Cabin` (trop de NaN : supprimer ou en tirer une info) ? `Embarked` (2 NaN seulement) ? Y a-t-il des doublons ?

### Étape 5 — Feature engineering

Crée de nouvelles variables pertinentes. Quelques pistes à étudier (choisis et justifie) :

- **Titre** extrait de `Name` (regex : `str.extract(r" ([A-Za-z]+)\.")`), puis regroupe les titres rares.
- **Taille de la famille** : `SibSp + Parch + 1`.
- **Voyage seul** (booléen).
- **Tranches d'âge** (`pd.cut`).
- **A une cabine** (booléen : `Cabin` non nul).
- Peut-être le logarithme du tarif (`np.log1p`) pour réduire l'asymétrie.
- Question : pourquoi ces variables pourraient-elles aider le modèle ? Laquelle est la plus prometteuse selon ton EDA ?

### Étape 6 — Préparation pour le modèle

- Outils : `train_test_split(stratify=y)`, `ColumnTransformer`, `Pipeline`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`.
- ⚠️ Mets **tout** le prétraitement dans le pipeline pour éviter la fuite de données.
- Questions : quelles colonnes sont numériques, lesquelles catégorielles ? Quelles colonnes retires-tu (`Ticket`, `Name`, `PassengerId`) et pourquoi ?

### Étape 7 — Modélisation

1. **Baseline** : `DummyClassifier(strategy="most_frequent")`. Note ses scores.
2. **Régression logistique** (baseline sérieuse, explicable).
3. **Arbre de décision** limité en profondeur (très explicable, tu peux le dessiner avec `plot_tree`).
4. **Forêt aléatoire**.
5. **Gradient boosting** (`HistGradientBoostingClassifier`).
6. (Optionnel) KNN, SVM, MLP.

- Compare-les par **validation croisée stratifiée à 5 plis** sur le train. Métriques : accuracy, **précision**, rappel, F1, ROC-AUC.
- Rassemble les résultats dans **un tableau pandas** (modèle × métrique, moyenne ± écart-type).

### Étape 8 — Optimisation et évaluation finale

- `GridSearchCV` (ou `RandomizedSearchCV`) sur les 1 ou 2 meilleurs modèles.
- Choisis ton modèle **en tenant compte de la contrainte du musée** (explicabilité + précision).
- Évalue-le **une seule fois** sur le jeu de test : matrice de confusion, `classification_report`, courbe ROC.
- Trace la **courbe d'apprentissage** : y a-t-il du surapprentissage ?
- Pour un réglage fin du seuil de décision, regarde `precision_recall_curve`.

### Étape 9 — Interprétation et présentation

- Importance des variables (`feature_importances_` ou `permutation_importance`, coefficients de la régression logistique).
- Analyse des erreurs : quels profils sont les plus mal prédits ? Pourquoi ?
- Teste ton modèle sur 3 profils fictifs (ex. : une enfant de 8 ans en 3ᵉ classe, un homme de 40 ans en 1ʳᵉ classe, une femme de 30 ans en 2ᵉ classe) et affiche les probabilités avec `predict_proba`.
- Sauvegarde le pipeline avec `joblib.dump`.
- Prépare ta présentation en suivant le **plan du chapitre 11** (9 sections).
- **Limites à discuter** : petit jeu de données (~900 lignes), données historiques, une corrélation n'est pas une causalité, un modèle qui reproduit les priorités d'évacuation de l'époque (femmes et enfants d'abord, classes supérieures favorisées) : question éthique si on généralise à d'autres contextes.

## 13.3 Grille d'auto-évaluation

| Critère                                              | ✅ |
| ----------------------------------------------------- | -- |
| Le problème et la métrique sont clairement définis |    |
| Le jeu de test n'a été utilisé qu'une seule fois   |    |
| Aucune fuite de données (pipeline)                   |    |
| Une baseline est comparée aux modèles               |    |
| La validation croisée est utilisée                  |    |
| Les métriques sont adaptées et interprétées       |    |
| Les variables importantes sont expliquées            |    |
| Les limites sont discutées                           |    |
| La présentation raconte une histoire                 |    |

## 13.4 Objectif de performance (repère)

Sur le Titanic, une bonne baseline atteint environ 77 à 80 % d'accuracy, et des modèles bien construits tournent généralement autour de 80 à 84 %. Ne cherche pas à gagner quelques points à tout prix : ce qui compte ici, c'est la **démarche propre**.

---

# 🧭 Plan de travail conseillé (pour ne pas te disperser)

| Jour  | Contenu                                         |
| ----- | ----------------------------------------------- |
| 1     | Chapitres 0-1 (NumPy) + exercice 1              |
| 2-3   | Chapitre 2 (Pandas) + exercice 2                |
| 4     | Chapitres 3-4 (graphiques) + exercice 3         |
| 5     | Chapitre 5 (SciPy) + exercice 4                 |
| 6     | Chapitres 6-7 (concepts + scikit-learn)         |
| 7-8   | Chapitres 8-9 (modèles)                        |
| 9     | Chapitre 10 (évaluation) + exercices 5-6       |
| 10    | Chapitres 11-12 (présentation + deep learning) |
| 11-12 | Chapitre 13 : projet Titanic                    |

> Adapte le rythme à ton planning, notamment à la date de ton hackathon. Si tu manques de temps, priorise : **Pandas → graphiques → chapitres 6, 7, 8, 10 → Titanic**. NumPy, SciPy et deep learning peuvent se voir en survol.

---

# 📚 Aide-mémoire final

```python
# Workflow ML type
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

X, y = df.drop(columns="cible"), df["cible"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
])
pipe = Pipeline([("prep", prep), ("clf", RandomForestClassifier(random_state=42))])

cv = StratifiedKFold(5, shuffle=True, random_state=42)
print(cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1").mean())

gs = GridSearchCV(pipe, {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 6, 10]},
                  cv=cv, scoring="f1", n_jobs=-1).fit(X_tr, y_tr)
print(classification_report(y_te, gs.predict(X_te)))
```

**Bonne route ! Quand tu termines un exercice, envoie-moi ton code : je le relis, je te montre où améliorer, et on passe au chapitre suivant. 🚀**
