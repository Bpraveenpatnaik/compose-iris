import os
import pickle
import pandas as pd
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

# define the class encodings and reverse encodings
classes = {0: "Red wine", 1: "White wine", 2: "Grape wine"}
r_classes = {y: x for x, y in classes.items()}

# function to train and load the model during startup
def init_model():
    if not os.path.isfile("models/data_wine.pkl"):
        clf = LogisticRegression()
        pickle.dump(clf, open("models/data_wine.pkl", "wb"))


# function to train and save the model as part of the feedback loop
def train_model(data):
    # load the model
    clf = pickle.load(open("models/data_wine.pkl", "rb"))

    # pull out the relevant X and y from the FeedbackIn object
    X = [list(d.dict().values())[:-1] for d in data]
    y = [r_classes[d.wine_class] for d in data]

    # fit the classifier again based on the new data obtained
    clf.fit(X, y)

    # save the model
    pickle.dump(clf, open("models/data_wine.pkl", "wb"))
