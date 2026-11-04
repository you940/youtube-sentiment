import os
import requests
import pandas as pd

URL = "https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv"
OUT = "data/raw/reddit.csv"

def main():
    os.makedirs("data/raw", exist_ok=True)
    r = requests.get(URL, timeout=60)
    r.raise_for_status()
    with open(OUT, "wb") as f:
        f.write(r.content)

    df = pd.read_csv(OUT)
    print("Colonnes :", list(df.columns))
    print("Nombre de commentaires :", len(df))
    print("Distribution des labels :")
    print(df.iloc[:, -1].value_counts())

if __name__ == "__main__":
    main()