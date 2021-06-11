# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import libraries

# %%
import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm as colormap
from tabulate import tabulate


# %%
panda.set_option("display.max_rows", 500)
panda.set_option("display.max_columns", 500)
panda.set_option("display.width", 1000)

colums = []
with open("../data/columns.txt", "r") as doc:
    lines = doc.readlines()
    for line in lines:
        colums.append(line.split(":")[0])


# %%
df = panda.read_csv("../data/spambase.data", names=colums)


# %%

print(tabulate(df.describe(), headers="keys", tablefmt="psql"))


# %%
print(tabulate(df.info(), headers="keys", tablefmt="psql"))


# %% [markdown]
# ## Check for missing values

# %%
print(df.isnull().sum())
# df.isna().any().any()


# %%
correlation = df.corr()
plt.figure(figsize=(11, 8))
matrix = np.triu(correlation, k=1)
ax = sns.heatmap(
    correlation,
    xticklabels=correlation.columns,
    yticklabels=correlation.columns,
    cmap="coolwarm",
    square=True,
    linewidths=0.1,
    mask=matrix,
)
ax.set(title="all variables correlation heatmap")
plt.show()


# %%
# top correlation with class (negativ and positiv)

k = 20  # number of variables
cols = correlation.abs().nlargest(k, "class")["class"].index
cm = np.corrcoef(df[cols].values.T)
plt.figure(figsize=(11, 8))
ax = sns.heatmap(
    cm,
    yticklabels=cols.values,
    xticklabels=cols.values,
    cmap="coolwarm",
    annot=True,
    square=True,
    fmt=".2f",
    mask=np.tri(k, k=-1).T,
)
ax.set(title=f"top {k} variables correlation with class heatmap")
plt.show()

# %% [markdown]
# Some correlations between the main class (spam, !spam) and the first 20 variables

# %%
sns.pairplot(data=df[cols[:5].values], hue="class")
plt.show()
# %% [markdown]
# ## Average word frequency in spam vs !spam

# %%
pivot_class_freq = panda.pivot_table(
    df,
    values=df.drop(
        [
            "class",
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ],
        axis=1,
    ),
    index="class",
    aggfunc="mean",
)
print(tabulate(pivot_class_freq, headers="keys", tablefmt="psql"))


# %%
plt.figure(figsize=(11, 8))
plt.xticks(rotation=70)
sns.barplot(
    x=[c.replace("char_", "").replace("word_", "") for c in pivot_class_freq.columns],
    y=pivot_class_freq.iloc[0] - pivot_class_freq.iloc[1],
).set(
    title="Non Spam minus Spam average word and char frequency",
    ylabel="Average frequency difference",
)
plt.show()

# %% [markdown]
# The words "you" and "your" are fare more frequent in spam emails than in !spam.
#
# The words "free", "george" and "hp" are fare more frequent in !spam emails than in spam.
#
# From the dataset description we know that the word "George" (and "650" as well, but is not significant) is not spam.
# So, the spammer don't know the victim name and call him "you" instead of his real name.

# %%
pivot_class_cap = panda.pivot_table(
    df,
    values=df[
        [
            "capital_run_length_average",
            "capital_run_length_longest",
            "capital_run_length_total",
        ]
    ],
    index="class",
)
print(tabulate(pivot_class_cap, headers="keys", tablefmt="psql"))


# %%
plt.xticks(rotation=10)
sns.barplot(
    x=pivot_class_cap.columns, y=pivot_class_cap.iloc[1] - pivot_class_cap.iloc[0]
).set(title="Spam minus !Spam", ylabel="average difference")
plt.show()
# %% [markdown]
# Spam emails have:
# - greater average length of uninterrupted sequences of capital letters;
# - greater length of longest uninterrupted sequence of capital letters;
# - greater total number of capital letters in the e-mail.
#
# Capital letters are far used in spam emails, the more frequent they are, the more probably is spam
#
# So, the spammer prefers to use capital letters to focus the victim attention on specific words, to scare him and rush him to click on the fake link in the email.

# %%

k = 10
df_dropped = df.drop(
    [
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
    ],
    axis=1,
)
df_dropped = (
    df_dropped[df_dropped["class"] == 1]
    .drop(["class"], axis=1)
    .sum()
    .sort_values(ascending=False)[:k]
)
plt.figure(figsize=(11, 8))
plt.pie(
    df_dropped,
    labels=df_dropped.index,
    autopct="%1.1f%%",
    explode=[0.05] * k,
    colors=colormap.get_cmap("Set1_r").colors,
)
plt.show()

# %% [markdown]
#
