# === Mushroom Classification: SVM + KNN + RandomForest (İstenen Çıktılar) ===

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Veri 
CSV_PATH = "mushrooms.csv"   
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Bulunamadı: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# Hedef: p=1 (poisonous), e=0 (edible)
y = (df["class"] == "p").astype(int).values
X = df.drop(columns=["class"])
cat_cols = X.columns.tolist()

# Ön işleme 
# SVM/KNN -> OneHot + Standardize; RF -> yalnızca OneHot
ohe_dist = ColumnTransformer(
    transformers=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
    remainder="drop"
)
ohe_tree = ColumnTransformer(
    transformers=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)],
    remainder="drop"
)

svm_clf = Pipeline(steps=[
    ("ohe", ohe_dist),
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=42)),
])

knn_clf = Pipeline(steps=[
    ("ohe", ohe_dist),
    ("scaler", StandardScaler(with_mean=False)),
    ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance")),
])

rf_clf = Pipeline(steps=[
    ("ohe", ohe_tree),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_split=5, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1
    )),
])

models = [("SVM (RBF)", svm_clf), ("KNN (k=5, distance)", knn_clf), ("Random Forest", rf_clf)]

# Train/Test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Değerlendirme 
def evaluate_model(name, pipe):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Sınıflandırma raporu
    print(f"\n[{name}] - Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    # Konfüzyon matrisi
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"{name} - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # ROC eğrisi
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, name=name)
    plt.title(f"{name} - ROC Curve")
    plt.tight_layout()
    plt.show()

    return {"Model": name, "Accuracy": acc}

# Çalıştır ve metrikleri topla
rows = []
for name, pipe in models:
    rows.append(evaluate_model(name, pipe))

# Accuracy Tablosu
acc_df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
acc_df["Accuracy"] = acc_df["Accuracy"].round(4)  # 4 basamaklı gösterim
print("\n=== Accuracy Tablosu (Sıralı) ===")
print(acc_df.to_string(index=False))


