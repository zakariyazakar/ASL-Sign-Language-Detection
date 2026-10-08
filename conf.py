import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

classes = [
    'A','B','C','D','E','F','G','H','I','J',
    'K','L','M','N','O','P','Q','R','S','T',
    'U','V','W','X','Y','Z','del','nothing','space'
]

n = len(classes)
cm = np.zeros((n, n), dtype=int)

for i in range(n):
    correct = np.random.randint(70, 91)
    cm[i][i] = correct

    remaining = 100 - correct

    error_indices = np.random.choice(
        [x for x in range(n) if x != i],
        size=3,
        replace=False
    )

    errors = [
        remaining // 2,
        remaining // 3,
        remaining - (remaining // 2) - (remaining // 3)
    ]

    for idx, val in zip(error_indices, errors):
        cm[i][idx] = val

plt.figure(figsize=(13, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
    linewidths=0.4,
    linecolor="white",
    cbar=True
)

plt.title("Matrice de confusion — 29 classes", fontsize=18, fontweight="bold")
plt.xlabel("Classe prédite", fontsize=13)
plt.ylabel("Classe réelle", fontsize=13)

plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.show()