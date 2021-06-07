# Spambase!

Spam classification using the spambase dataset (https://archive.ics.uci.edu/ml/datasets/spambase)

# Files

There are 2 main files (both ipynb and py):

- eda (exploration)
- model (classification)

## API

#### If you want to test the model prediction:

you have to clone the repo and install requirements.txt

> pip **install** -r **requirements**.txt

or

if you use **pipenv**, simply

> pipenv install

Then start the **server** :

> uvicorn main:app --reload

Next, open the browser at
http://127.0.0.1:8000/docs#/default/read_root__get
