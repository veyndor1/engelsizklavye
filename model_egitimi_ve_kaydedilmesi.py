import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open("data.pickle", "rb"))

# Elemanları aynı şekle getirerek NumPy dizisine dönüştürme
max_length = max(len(item) for item in data_dict["data"])
data = np.array([np.pad(item, (0, max_length - len(item))) for item in data_dict["data"]])

labels = np.asarray(data_dict["Labels"])

xtrain, xtest, ytrain, ytest = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(xtrain, ytrain)

y_pred = model.predict(xtest)

score = accuracy_score(y_pred, ytest)

print("Başari:", score * 100)

f = open("model.pickle", "wb")
pickle.dump({"model": model}, f)
f.close()