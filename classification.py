"""Practical 4: Classification Models (SVM, Decision Tree, KNN, Naive Bayes)
Dataset: Breast Cancer Wisconsin (Diagnostic) - built into scikit-learn."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, log_loss, confusion_matrix)

data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "SVM": Pipeline([("sc", StandardScaler()),
                     ("m", SVC(kernel="rbf", C=1.0, probability=True, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": Pipeline([("sc", StandardScaler()),
                     ("m", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
}

results, cms = {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = np.clip(model.predict_proba(X_test), 1e-15, 1 - 1e-15)
    results[name] = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "loss": log_loss(y_test, proba),
    }
    cms[name] = confusion_matrix(y_test, pred).tolist()
    print(name, {k: round(v, 4) for k, v in results[name].items()}, cms[name])

json.dump({"results": results, "cms": cms, "n_train": len(y_train), "n_test": len(y_test),
           "n_features": X.shape[1], "counts": np.bincount(y).tolist()},
          open("results.json", "w"), indent=1)

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
for ax, (name, cm) in zip(axes, cms.items()):
    cm = np.array(cm)
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_title(name); ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Malignant", "Benign"]); ax.set_yticklabels(["Malignant", "Benign"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout(); plt.savefig("confusion_matrices.png", dpi=160); plt.close()

# Metric comparison
names = list(results); metrics = ["accuracy", "precision", "recall", "f1"]
w = 0.2; x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 4.2))
for i, m in enumerate(metrics):
    ax.bar(x + i * w - 1.5 * w, [results[n][m] for n in names], w, label=m.capitalize())
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylim(0.85, 1.0)
ax.set_ylabel("Score"); ax.set_title("Performance comparison of classifiers")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False); ax.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("metrics_comparison.png", dpi=160); plt.close()
