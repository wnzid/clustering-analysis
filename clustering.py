import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

#loading dataset
df = pd.read_csv("dataset_039.csv")

target_column = "target"

X = df.drop(columns=[target_column])
y = df[target_column]

output_dir = "clustering_results"
os.makedirs(output_dir, exist_ok=True)

print("=" * 70)
print("DATASET INFO")
print("=" * 70)
print("Dataset shape:", df.shape)
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
print("Target column:", target_column)
print("Unique classes:", sorted(y.unique()))
print("Number of classes:", y.nunique())
print()

#strandardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("=" * 70)
print("PCA INFO")
print("=" * 70)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total explained variance:", pca.explained_variance_ratio_.sum())
print()

#helpers
def make_composition_table(true_labels, cluster_labels):
    return pd.crosstab(
        pd.Series(true_labels, name="True Class"),
        pd.Series(cluster_labels, name="Cluster")
    )

def plot_pca_clusters(X_2d, labels, title, save_name):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=35,
        alpha=0.8,
        edgecolors="k",
        linewidths=0.2
    )
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)
    plt.colorbar(scatter, label="Cluster / Class")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {save_path}")
    plt.close()

def evaluate_clustering(name, true_labels, cluster_labels, X_for_silhouette):
    n_clusters_found = len(np.unique(cluster_labels))
    ari = adjusted_rand_score(true_labels, cluster_labels)

    if n_clusters_found > 1:
        sil = silhouette_score(X_for_silhouette, cluster_labels)
    else:
        sil = None

    print("=" * 70)
    print(name)
    print("=" * 70)
    print("Clusters found:", n_clusters_found)
    print("Adjusted Rand Index (ARI):", round(ari, 4))
    if sil is not None:
        print("Silhouette Score:", round(sil, 4))
    else:
        print("Silhouette Score: not available (only one cluster)")
    print()

    table = make_composition_table(true_labels, cluster_labels)
    print("Cluster composition table:")
    print(table)
    print()

    return {
        "name": name,
        "n_clusters": n_clusters_found,
        "ari": ari,
        "silhouette": sil,
        "table": table
    }

#true label PCA
plot_pca_clusters(
    X_pca,
    y,
    "True Classes in PCA Space",
    "01_true_classes_pca.png"
)

true_table = pd.DataFrame({
    "Class": sorted(y.unique()),
    "Count": y.value_counts().sort_index().values
})

#K-means with k = number of true classes
n_classes = y.nunique()

kmeans_4 = KMeans(n_clusters=n_classes, random_state=42, n_init=20)
kmeans_4_labels = kmeans_4.fit_predict(X_scaled)

result_k4 = evaluate_clustering(
    f"K-Means with k = {n_classes}",
    y,
    kmeans_4_labels,
    X_scaled
)

plot_pca_clusters(
    X_pca,
    kmeans_4_labels,
    f"K-Means Clustering (k = {n_classes}) in PCA Space",
    "02_kmeans_k4_pca.png"
)

#k-means with k = number of true classes + 2
k_more = n_classes + 2

kmeans_6 = KMeans(n_clusters=k_more, random_state=42, n_init=20)
kmeans_6_labels = kmeans_6.fit_predict(X_scaled)

result_k6 = evaluate_clustering(
    f"K-Means with k = {k_more}",
    y,
    kmeans_6_labels,
    X_scaled
)

plot_pca_clusters(
    X_pca,
    kmeans_6_labels,
    f"K-Means Clustering (k = {k_more}) in PCA Space",
    "03_kmeans_k6_pca.png"
)

#agglomerative clustering
agglomerative_settings = [
    {"linkage": "ward", "distance_threshold": 25},
    {"linkage": "ward", "distance_threshold": 30},
    {"linkage": "ward", "distance_threshold": 35},
    {"linkage": "ward", "distance_threshold": 40},
    {"linkage": "complete", "distance_threshold": 15},
    {"linkage": "complete", "distance_threshold": 20},
    {"linkage": "average", "distance_threshold": 10},
    {"linkage": "average", "distance_threshold": 15},
]

agg_summary_rows = []
best_result = None
best_score = -999999

for params in agglomerative_settings:
    linkage = params["linkage"]
    threshold = params["distance_threshold"]

    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage=linkage
    )

    agg_labels = model.fit_predict(X_scaled)

    result = evaluate_clustering(
        f"Agglomerative | linkage={linkage}, threshold={threshold}",
        y,
        agg_labels,
        X_scaled
    )

    agg_summary_rows.append({
        "Method": "Agglomerative",
        "Linkage": linkage,
        "Distance_Threshold": threshold,
        "Clusters_Found": result["n_clusters"],
        "ARI": round(result["ari"], 4),
        "Silhouette": round(result["silhouette"], 4) if result["silhouette"] is not None else None
    })

    score = -abs(result["n_clusters"] - n_classes) + result["ari"]

    if score > best_score:
        best_score = score
        best_result = {
            "linkage": linkage,
            "threshold": threshold,
            "labels": agg_labels,
            "table": result["table"],
            "n_clusters": result["n_clusters"],
            "ari": result["ari"],
            "silhouette": result["silhouette"]
        }

agg_summary_df = pd.DataFrame(agg_summary_rows)

print("=" * 70)
print("BEST AGGLOMERATIVE RESULT")
print("=" * 70)
print("Best linkage:", best_result["linkage"])
print("Best threshold:", best_result["threshold"])
print("Clusters found:", best_result["n_clusters"])
print("ARI:", round(best_result["ari"], 4))
if best_result["silhouette"] is not None:
    print("Silhouette:", round(best_result["silhouette"], 4))
else:
    print("Silhouette: not available")
print()

print("Best agglomerative composition table:")
print(best_result["table"])
print()

plot_pca_clusters(
    X_pca,
    best_result["labels"],
    f"Best Agglomerative Clustering ({best_result['linkage']}, threshold={best_result['threshold']})",
    "04_best_agglomerative_pca.png"
)

#summary table
summary_df = pd.DataFrame([
    {
        "Method": "K-Means",
        "Setting": f"k={n_classes}",
        "Clusters_Found": result_k4["n_clusters"],
        "ARI": round(result_k4["ari"], 4),
        "Silhouette": round(result_k4["silhouette"], 4) if result_k4["silhouette"] is not None else None
    },
    {
        "Method": "K-Means",
        "Setting": f"k={k_more}",
        "Clusters_Found": result_k6["n_clusters"],
        "ARI": round(result_k6["ari"], 4),
        "Silhouette": round(result_k6["silhouette"], 4) if result_k6["silhouette"] is not None else None
    },
    {
        "Method": "Agglomerative",
        "Setting": f"linkage={best_result['linkage']}, threshold={best_result['threshold']}",
        "Clusters_Found": best_result["n_clusters"],
        "ARI": round(best_result["ari"], 4),
        "Silhouette": round(best_result["silhouette"], 4) if best_result["silhouette"] is not None else None
    }
])

#save all in one excel
excel_path = os.path.join(output_dir, "clustering_results.xlsx")

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    true_table.to_excel(writer, sheet_name="Class_Counts", index=False)
    result_k4["table"].to_excel(writer, sheet_name="KMeans_k4")
    result_k6["table"].to_excel(writer, sheet_name="KMeans_k6")
    agg_summary_df.to_excel(writer, sheet_name="Agglomerative_Summary", index=False)
    best_result["table"].to_excel(writer, sheet_name="Best_Agglomerative")
    summary_df.to_excel(writer, sheet_name="Final_Summary", index=False)

print(f"Saved Excel workbook: {excel_path}")

print("=" * 70)
print("DONE")
print("=" * 70)
print(f"Plots saved in folder: {output_dir}")
print("Excel workbook contains all important tables.")