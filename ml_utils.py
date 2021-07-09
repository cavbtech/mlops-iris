from os import X_OK
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

clf = GaussianNB()

classes = {
    0: "Iris Setosa",
    1: "Iris Versicolour",
    2: "Iris Virginica",
	3: "CAVB Flower"
}

def load_model():
	X, y = datasets.load_iris(return_X_y=True)

	X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
	clf.fit(X_train, y_train)

	acc = accuracy_score(y_test, clf.predict(X_test))
	print(f"Model trained with accuracy: {round(acc, 3)}")

def predict(query_data):
	print(f"query_data.dict().values()={query_data.dict().values()}")
	x = list(query_data.dict().values())
	prediction = clf.predict([x])[0] 
	print(f"Model prediction: {classes[prediction]}")
	return classes[prediction]

# def retrain(data):
# 	X =	[list(d.dict().values())[:-1] for d in data]
# 	y = [classes[d.flower_class] for d in data]
# 	print(X)
# 	print(y)
# 	clf.fit(X,y)

# def retrain_single(data):
# 	X =	np.array[data].reshape(-1,1)

# 	y = classes[data.flower_class]
# 	print(X)
# 	print(y)
# 	clf.fit(X,y)




