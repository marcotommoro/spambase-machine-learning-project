# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Import libraries

# %%
import pandas as panda
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from sklearn import linear_model
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    plot_confusion_matrix,
)

from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tabulate import tabulate

# %%
panda.set_option("display.max_columns", 57)

columns = []
with open("columns.txt", "r") as doc:
    lines = doc.readlines()
    for line in lines:
        columns.append(line.split(":")[0])


# %%
df = panda.read_csv("spambase.data", header=None)

# %% [markdown]
# ## Data preprocessing. Training and test set

# %%
y = df[np.shape(df)[1] - 1]  # class column
x = df.drop([np.shape(df)[1] - 1], axis=1)  # remove class column
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=69, train_size=0.66
)

# %% [markdown]
# ## Initialize models

# %%
MODELS = []
MODELS.append(("Logistig Regression", LogisticRegression(max_iter=10000)))
MODELS.append(("Decision Tree Classifier", DecisionTreeClassifier()))
MODELS.append(("SVC", SVC()))
MODELS.append(("Naive Bayes", GaussianNB()))
MODELS.append(("K Nearest Neighbour", KNeighborsClassifier()))
MODELS.append(("Support Vector Classification", SVC()))
MODELS.append(("Stochastic Gradient Descent", SGDClassifier()))
MODELS.append(("Linear Discriminant Analysis", LinearDiscriminantAnalysis()))
MODELS.append(("Gradient Boosting Classification ", GradientBoostingClassifier()))
MODELS.append(("Random Forest Classification", RandomForestClassifier()))


# %%
def get_stats(y_predicted):
    accuracy = accuracy_score(y_test, y_predicted)
    precision = precision_score(y_test, y_predicted)
    recall = recall_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted)
    roc = roc_auc_score(y_test, y_predicted)
    return [accuracy, precision, recall, f1, roc]


# %% [markdown]
# ### Fit train data in model and predict

# %%
column_names = ["Model", "Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
results = []
for name, model in MODELS:
    model.fit(X=x_train, y=y_train)
    y_predicted = model.predict(x_test)
    results.append([name, *get_stats(y_predicted)])

# %% [markdown]
# ## Create dataframe and sort

# %%
res = panda.DataFrame(results, columns=column_names)
res = res.sort_values(by="Precision", ascending=False)

print(tabulate(res, headers="keys", tablefmt="psql"))

# %% [markdown]
# ### Sort the initial models by precision

# %%
diz = {v: i for i, v in enumerate(list(res["Model"]))}
SORTED_MODELS = sorted(MODELS, key=lambda x: diz[x[0]])

# %%
items_per_row = 3
fig, axs = plt.subplots(
    ceil(len(SORTED_MODELS) / items_per_row),
    items_per_row,
    sharex=True,
    figsize=(11, 8),
)
fig.suptitle("Confusion Matrix")
fig.tight_layout(pad=2.0, h_pad=4.0)

for i, (name, model) in enumerate(SORTED_MODELS):
    y, x = i // items_per_row, i % items_per_row
    plot_confusion_matrix(model, x_test, y_test, ax=axs[y, x])
    axs[y, x].set_title(name)

for i in range(
    len(SORTED_MODELS), ceil(len(SORTED_MODELS) / items_per_row) * items_per_row
):
    y, x = i // items_per_row, i % items_per_row
    axs[y, x].axis("off")

plt.show()

# %%
