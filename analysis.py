import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../results/similarity_scores.csv")

# Figure 1
plt.figure()
df["similarity"].plot(kind="hist", bins=10, density=True)
df["similarity"].plot(kind="kde")
plt.title("Similarity Distribution")
plt.savefig("../figures/fig1.png")
plt.close()

# Figure 2
plt.figure()
df.boxplot(column="similarity", by="prompt_type")
plt.suptitle("")
plt.title("Similarity by Prompt Type")
plt.savefig("../figures/fig2.png")
plt.close()

# Figure 3
means = df.groupby("prompt_type")["similarity"].mean()
stds = df.groupby("prompt_type")["similarity"].std()

plt.figure()
plt.errorbar(means.index, means.values, yerr=stds.values, fmt='o')
plt.title("Mean Similarity with Variability")
plt.savefig("../figures/fig3.png")
plt.close()
