import pickle
#from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
# define a Gaussain NB classifier
clf = LogisticRegression()

# define the class encodings and reverse encodings
classes = {0: "Red wine", 1: "White wine", 2: "Grape wine"}
r_classes = {y: x for x, y in classes.items()}


# function to load the model
def load_model():
    global clf
    clf = pickle.load(open("models/data_wine.pkl", "rb"))


# function to predict the wine using the model
def predict(query_data):
    x = list(query_data.dict().values())
    prediction = clf.predict([x])[0]
    return classes[prediction]
