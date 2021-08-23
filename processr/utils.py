import os
import pickle
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# define the class encodings and reverse encodings
classes = {0: "Red wine", 1: "White wine", 2: "Grape wine"}
r_classes = {y: x for x, y in classes.items()}

# function to process data and return it in correct format
def process_data(data):
    processed = [
        {
            #"sepal_length": d.sepal_length,
            #"sepal_width": d.sepal_length,
            #"petal_length": d.petal_length,
            #"petal_width": d.petal_width,
            #"flower_class": d.flower_class,

            "alcohol": d.alcohol,
            "malic_acid": d.malic_acid,
            "ash": d.ash,
            "alcalinity_of_ash": d.alcalinity_of_ash,
            "wine_class": d.wine_class,
        }
        for d in data
    ]

    return processed
