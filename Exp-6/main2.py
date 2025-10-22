#this main file is for wine dataset
import numpy as np
from data2 import load_wine_dataset
from sklearn.preprocessing import StandardScaler
from utils import train_test_split
from knn_classifier import KNN_classifier
import matplotlib.pyplot as plt

X, y = load_wine_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#scaling features as it contains large values in decimal
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

knn = KNN_classifier(k=3)
knn.fit(X_train_scaled, y_train)
y_pred = knn.predict(X_test_scaled)

knn2 = KNN_classifier(k=3)
knn2.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)

#calculating accuracy
accuracy = np.sum(y_pred == y_test)/ len(y_test)
accuracy2 = np.sum(y_pred2 == y_test)/ len(y_test)
print("Accuracy of wine dataset at k = 3 with scaling: ", accuracy)
print("Accuracy of wine dataset at k = 3 without scaling: ", accuracy2)

#now checking for different values of k
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

for k in k_values:
    knn = KNN_classifier(k=k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    acc = np.sum(y_pred == y_test) / len(y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k value for wine dataset')
plt.show()