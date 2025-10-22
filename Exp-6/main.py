from utils import train_test_split
from data import load_preprocess_data
from knn_classifier import KNN_classifier
import numpy as np
import matplotlib.pyplot as plt

X, y = load_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

print("Shapes:")
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

knn = KNN_classifier(k=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

#checking the result by comparing y_pred and y_test
accuracy = np.sum(y_pred == y_test)/len(y_test)
print(f"Accuracy when k = 3: {accuracy:.2f}")

#now checking for different values of k
k_values = [1, 3, 5, 7, 9, 11, 15]
accuracies = []

for k in k_values:
    knn = KNN_classifier(k=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = np.sum(y_pred == y_test) / len(y_test)
    accuracies.append(acc)

plt.plot(k_values, accuracies, marker='o')
plt.xlabel('k value')
plt.ylabel('Accuracy')
plt.title('Accuracy vs k value')
plt.show()

