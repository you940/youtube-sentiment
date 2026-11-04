import re
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r"http\S+|www\.\S+", " ", t)      # URLs
    t = re.sub(r"@\w+", " ", t)                  # mentions
    t = re.sub(r"[^a-z0-9\s'!?]", " ", t)        # caractères spéciaux
    return re.sub(r"\s+", " ", t).strip()

def main():
    df = pd.read_csv("data/raw/reddit.csv")
    df = df.rename(columns={"clean_comment": "text", "category": "label"})
    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates()

    print("Nombre de commentaires après nettoyage :", len(df))
    print("Distribution des classes (%) :")
    print(df["label"].value_counts(normalize=True) * 100)
    print("Longueur des textes (en mots) :")
    print(df["text"].str.split().str.len().describe())

    os.makedirs("data/processed", exist_ok=True)
    train, test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    df.to_csv("data/processed/clean.csv", index=False)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    print("Train :", len(train), "| Test :", len(test))

if __name__ == "__main__":
    main()