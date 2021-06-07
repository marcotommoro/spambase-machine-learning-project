from typing import Optional
from fastapi import FastAPI

import pandas as panda
from pandas.core.algorithms import mode
import numpy as np
import matplotlib.pyplot as plt
from math import ceil

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import re

app = FastAPI()

columns = []
with open("columns.txt", "r") as doc:
    lines = doc.readlines()
    for line in lines:
        columns.append(line.split(":")[0])

df = panda.read_csv("spambase.data", header=None)

y = df[np.shape(df)[1] - 1]  # class column
x = df.drop([np.shape(df)[1] - 1], axis=1)  # remove class column
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=69, train_size=0.66
)
model = RandomForestClassifier()
model = model.fit(X=x_train, y=y_train)


def predict(email):
    v = process_email(email)
    v = panda.DataFrame([v])
    y_predicted = model.predict(v)
    print(y_predicted)
    print("ciaoooo")
    return y_predicted


def process_email(email):
    vect = []
    nb_word = len(re.findall(r"\w+", email))

    # word/char freq
    email_low = email.lower()
    for col in columns[:-4]:  # the last tree dont look for word/char freq
        # the columns names a in the format : word/char_freq_ref
        ref = col.split("_")[2]
        if ref in ["(", "["]:
            ref = "\\" + ref
        match_count = len(re.findall(f"({ref})", email_low))

        vect.append(100 * match_count / nb_word)
    # last 3 variables
    # every sentence in capital letters
    capital = re.findall("[A-Z, ,\d,\,]{2,}", email)
    longest = 0
    sum_cap = 0
    if len(capital) != 0:
        for match in capital:
            sum_cap += len(
                match.replace(" ", "")
            )  # removing " " to get the total number of capital letter
            size = len(match)
            if size > longest:
                longest = size
    else:
        capital = [1]
    vect.append(
        sum_cap / len(capital)
    )  # average length of uninterrupted sequences of capital letters
    vect.append(longest)  # length of longest uninterrupted sequence of capital letters
    vect.append(sum_cap)  # total number of capital letters in the e-mail

    return vect


@app.get("/")
def read_root(email: str):
    nostradamus = predict(email)
    return {"spam": bool(nostradamus)}
