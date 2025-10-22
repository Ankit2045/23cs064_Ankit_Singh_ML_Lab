import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
def load_wine_dataset():
    #importing the wine dataset
    wine = fetch_ucirepo(id = 109)
    #getting features and targets
    X = pd.DataFrame(wine.data.features)
    y = pd.DataFrame(wine.data.targets)

    #converting X and y to numpy arrays
    X = X.to_numpy()
    y = y.to_numpy().ravel() #ravel to create a single array
    return X, y




