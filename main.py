import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

data = pd.read_csv('data/A_Z Handwritten Data.csv')

# Data exploration and preparation:
# Identify the number of unique classes and show their distribution.
first_column = data.iloc[:, 0]
print(data.iloc[:, 0].nunique())
print(data.iloc[:, 0].value_counts())

# Normalize each image
images = data.drop(columns=0).values

# Normalize the images
images = images / 255.0

# Verify normalization
print(f'Max pixel value: {images.max()}')
print(f'Min pixel value: {images.min()}')