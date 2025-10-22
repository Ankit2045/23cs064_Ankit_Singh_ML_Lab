import matplotlib.pyplot as plt
import numpy as np
from data import X, y

species = np.unique(y)
colors = ['red', 'green', 'blue']

plt.figure(figsize=(12,10))

num_features = X.shape[1]
feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
plot_num = 1
for i in range(num_features):
    for j in range(i+1, num_features):
        plt.subplot(2,3, plot_num)
        for species_name, color in zip(species, colors):
            plt.scatter(
                X[y==species_name, i],
                X[y == species_name, j],
                color = color,
                label = species_name
            )
        plt.xlabel(feature_names[i])
        plt.ylabel(feature_names[j])
        plt.title(f'{feature_names[i]} vs {feature_names[j]}')
        plt.legend()
        plot_num += 1

plt.tight_layout()
plt.show()

