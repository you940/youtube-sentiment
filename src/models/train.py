import os
import time
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1) Chargement des données
train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")
ytr, yte = train["label"], test["label"]

# 2) Vectorisation TF-IDF
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2),
                      min_df=2, sublinear_tf=True)
Xtr = vec.fit_transform(train["text"])
Xte = vec.transform(test["text"])
print("Taille du vocabulaire :", len(vec.vocabulary_))

# 3) Logistic Regression + GridSearchCV
print("\n--- GridSearchCV (Logistic Regression) ---")
grid = GridSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced"),
    {"C": [1, 5, 10, 20, 50]},
    cv=3, scoring="f1_macro", n_jobs=-1, verbose=1
)
grid.fit(Xtr, ytr)
print("Meilleurs paramètres :", grid.best_params_)
best_lr = grid.best_estimator_

# 4) Comparaison avec d'autres algorithmes
print("\n--- Comparaison des modèles ---")
candidates = {
    "LogisticRegression (optimisé)": best_lr,
    "LinearSVC": LinearSVC(class_weight="balanced", C=0.5).fit(Xtr, ytr),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        n_jobs=-1, random_state=42).fit(Xtr, ytr),
}
for name, m in candidates.items():
    p = m.predict(Xte)
    print(f"{name:32s} accuracy={accuracy_score(yte, p):.4f}  "
          f"F1 macro={f1_score(yte, p, average='macro'):.4f}")

# 5) Évaluation détaillée du modèle retenu (Logistic Regression)
pred = best_lr.predict(Xte)
print("\n--- Rapport de classification (Logistic Regression) ---")
print(classification_report(yte, pred, digits=3))
print("Accuracy :", round(accuracy_score(yte, pred), 4))

# 6) Temps d'inférence pour un batch de 50 commentaires
batch = test["text"].head(50).tolist()
times = []
for _ in range(20):
    t0 = time.perf_counter()
    best_lr.predict_proba(vec.transform(batch))
    times.append((time.perf_counter() - t0) * 1000)
print(f"Temps d'inférence (50 commentaires) : {sum(times)/len(times):.1f} ms en moyenne")

# 7) Matrice de confusion
cm = confusion_matrix(yte, pred, labels=[-1, 0, 1])
ConfusionMatrixDisplay(cm, display_labels=["Négatif", "Neutre", "Positif"]).plot(cmap="Blues")
plt.title("Matrice de confusion - Logistic Regression")
plt.savefig("reports/confusion_matrix.png", dpi=150)

# 8) Sauvegarde
joblib.dump(best_lr, "models/model.joblib")
joblib.dump(vec, "models/vectorizer.joblib")
print("\nModèle et vectoriseur sauvegardés dans models/")