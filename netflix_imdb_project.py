"""Extracted (and lightly cleaned) code from the PDF project:
'Netflix IMDB Score' (Data Mining project).

Notes:
- The original report appears to have been written in a notebook (Kaggle-style path).
- Some strings/quotes were fixed due to OCR artifacts.
- Update CSV_PATH to the location of your dataset file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


# -----------------------------
# 1) Load data
# -----------------------------
CSV_PATH = "/kaggle/input/netflix-imdb-scores/Netflix TV Shows and Movies.csv"  # TODO: change this path
df = pd.read_csv(CSV_PATH)

# Display the first few rows
print(df.head())

# Summary statistics + schema
print(df.describe())
print(df.info())

# Missing values per column
print("Missing values per column:")
print(df.isna().sum())


# -----------------------------
# 2) EDA plots
# -----------------------------
# Distribution of IMDb scores
plt.figure(figsize=(14, 8))
sns.histplot(df["imdb_score"], kde=True)
plt.xlabel("IMDb Score")
plt.ylabel("Frequency")
plt.title("Distribution of IMDb Scores")
plt.show()

# Count of different types (MOVIE, SHOW)
sns.countplot(x="type", data=df)
plt.xlabel("Type")
plt.ylabel("Count")
plt.title("Count of Different Types")
plt.show()

# Count of age certifications
sns.countplot(x="age_certification", data=df)
plt.xlabel("Age Certification")
plt.ylabel("Count")
plt.title("Count of Age Certifications")
plt.show()

# Relationship between IMDb score and runtime
sns.scatterplot(x="runtime", y="imdb_score", data=df)
plt.xlabel("Runtime")
plt.ylabel("IMDb Score")
plt.title("IMDb Score vs Runtime")
plt.show()

# Age certification distribution by type
plt.figure(figsize=(18, 6))
sns.countplot(x="age_certification", hue="type", data=df)
plt.xlabel("Age Certification")
plt.ylabel("Count")
plt.title("Age Certification Distribution by Type")
plt.legend(title="Type")
plt.show()


# -----------------------------
# 3) Missing-value replacement + correlations
# -----------------------------
# Replace NaN in age_certification with 'for all' (as mentioned in the report)
df.replace(to_replace=np.nan, value="for all", inplace=True)

# Correlation between imdb_score and imdb_votes
x = df["imdb_score"]
y = df["imdb_votes"]
correlation = x.corr(y)
print("Correlation (imdb_score vs imdb_votes):", correlation)

# Scatter for imdb_score vs runtime (matplotlib version used in report)
plt.scatter(df["imdb_score"], df["runtime"], c="blue", marker="*")
plt.ylabel("runtime")
plt.xlabel("imdb_score")
plt.title("correlation runtime & imdb")
plt.show()


# -----------------------------
# 4) Clustering (KMeans) + Plotly visualizations
# -----------------------------
# (Report uses runtime, release_year, imdb_score and tries KMeans with 5 clusters.)
X_cluster = df[["runtime", "release_year", "imdb_score"]].copy()

km = KMeans(n_clusters=5, random_state=42)
df["cluster"] = km.fit_predict(X_cluster).astype("category")

# 3D scatter (plotly)
px.scatter_3d(
    df,
    x="release_year",
    y="imdb_score",
    z="runtime",
    color="cluster",
).show()

# Example with different number of clusters (2 clusters) if needed:
# km2 = KMeans(n_clusters=2, random_state=42)
# df["cluster2"] = km2.fit_predict(X_cluster).astype("category")
# px.scatter_3d(df, x="release_year", y="imdb_score", z="runtime", color="cluster2").show()

# 2D scatter by cluster
px.scatter(
    df,
    x="imdb_score",
    y="runtime",
    color="cluster",
).show()


# -----------------------------
# 5) KNN preparation + training (as in report)
# -----------------------------
# Drop any missing values
df_knn = df.copy()
df_knn.dropna(inplace=True)

# Encode categorical features (object columns) using category codes
for column in df_knn.select_dtypes(include="object"):
    df_knn[column] = df_knn[column].astype("category")
    df_knn[column] = df_knn[column].cat.codes

# Split into features and labels (label was imdb_score in the report)
X = df_knn.drop("imdb_score", axis=1)
y = df_knn["imdb_score"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train KNN (n_neighbors=5)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_scaled, y)

# Predict (report predicts on the same X_scaled)
y_pred = knn.predict(X_scaled)
print("KNN params:", knn.get_params())


# -----------------------------
# 6) Manual confusion-matrix style evaluation (PG-13 thresholding)
# -----------------------------
# NOTE: This evaluation is based on a custom rule from the report:
# imdb_score >= 7.5 and age_certification == 'PG-13' considered "positive".

true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(df)):
    if df.loc[i, "imdb_score"] >= 7.5 and df.loc[i, "age_certification"] == "PG-13":
        true_positives += 1
    elif df.loc[i, "imdb_score"] < 7.5 and df.loc[i, "age_certification"] == "PG-13":
        false_negatives += 1
    elif df.loc[i, "imdb_score"] >= 7.5 and df.loc[i, "age_certification"] != "PG-13":
        false_positives += 1
    else:
        true_negatives += 1

print("Confusion matrix:")
print([[true_positives, false_positives], [false_negatives, true_negatives]])

# accuracy
accuracy = (true_positives + true_negatives) / len(df)
# sensitivity
sensitivity = (
    true_positives / (true_positives + false_negatives)
    if (true_positives + false_negatives)
    else 0.0
)
# specificity
specificity = (
    true_negatives / (true_negatives + false_positives)
    if (true_negatives + false_positives)
    else 0.0
)

print("Accuracy:", accuracy)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)