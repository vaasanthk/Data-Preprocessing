# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Vasanth\Desktop\Python program\Machine Learning A-Z\Part 1 - Data Preprocessing\Section 2  Part 1 - Data Preprocessing --\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Spliting the dataset into Training set and Teat set. 
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
