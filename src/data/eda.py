import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/processed/clean.csv")

sns.countplot(x="label", data=df)
plt.title("Distribution des classes")
plt.savefig("reports/class_distribution.png", dpi=150)
plt.clf()

df["len"] = df["text"].str.split().str.len()
sns.histplot(df["len"].clip(upper=200), bins=50)
plt.title("Longueur des textes (mots, tronquée à 200)")
plt.savefig("reports/text_length.png", dpi=150)
print("Graphiques enregistrés dans reports/")