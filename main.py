import numpy
import pandas
import matplotlib.pyplot
import sklearn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
import tensorflow
from tensorflow.keras.layers import Dense

# Data exploration and preparation
# Data loading
data = pandas.read_csv('data/A_Z Handwritten Data.csv')

# Identify the number of unique classes and show their distribution
print("Number of unique values:", data.iloc[:, 0].nunique(), "\n")
print("Data distribution: \n", data.iloc[:, 0].value_counts())

# Normalize each image
images = data.iloc[:, 1:].values
images = images / 255.0

# Split the data into training and testing datasets
features = images
targets = data.iloc[:, 0]
features, targets = sklearn.utils.shuffle(features, targets)

size_of_training = int(len(data) * 0.8)

features_training = features[:size_of_training]
features_testing = features[size_of_training:]

targets_training = targets[:size_of_training]
targets_testing = targets[size_of_training:]

# First experiment (You can use scikit-learn):
# Train SVM with a linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(features_training, targets_training)
targets_pred_linear = svm_linear.predict(features_testing)
conf_matrix_linear = confusion_matrix(targets_testing, targets_pred_linear)
f1_score_linear = f1_score(targets_testing, targets_pred_linear, average='weighted')
print("Confusion Matrix (Linear Kernel):\n", conf_matrix_linear)
print("Average F1 Score (Linear Kernel):", f1_score_linear, "\n")

# Train SVM with a non-linear kernel
svm_non_linear = SVC(kernel='rbf')
svm_non_linear.fit(features_training, targets_training)
targets_pred_non_linear = svm_non_linear.predict(features_testing)
conf_matrix_non_linear = confusion_matrix(targets_testing, targets_pred_non_linear)
f1_score_non_linear = f1_score(targets_testing, targets_pred_non_linear, average='weighted')
print("Confusion Matrix (Non Linear Kernel):\n", conf_matrix_non_linear)
print("Average F1 Score (Non Linear Kernel):", f1_score_non_linear)

# Split the training dataset into training and validation datasets.
size_of_validation = int(len(features_training) * 0.8)

features_train = features[:size_of_validation]
features_valid = features[size_of_validation:]

targets_train = targets[:size_of_validation]
targets_valid = targets[size_of_validation:]

# Third experiment (You can use TensorFlow):
# Design 2 Neural Networks (with different number of hidden layers, neurons, activations, etc.)
# First Neural Network: Simple architecture with one hidden layer
model1 = tensorflow.keras.models.Sequential()
model1.add(Dense(128, input_dim=784, activation='relu'))  # 128 neurons, input layer size 784 (28x28 pixels)
model1.add(Dense(26, activation='softmax'))  # 26 output neurons (A-Z)

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history1 = model1.fit(features_training, targets_train, epochs=10, batch_size=32, validation_data=(features_valid, targets_valid))

# Second Neural Network: More complex with two hidden layers
model2 = tensorflow.keras.models.Sequential()
model2.add(Dense(256, input_dim=784, activation='relu'))  # 256 neurons in first hidden layer
model2.add(Dense(128, activation='relu'))  # 128 neurons in second hidden layer
model2.add(Dense(26, activation='softmax'))  # 26 output neurons (A-Z)

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history2 = model2.fit(features_training, targets_train, epochs=10, batch_size=32, validation_data=(features_valid, targets_valid))

# Evaluate the models on the test set

# Model 1 evaluation
test_loss1, test_accuracy1 = model1.evaluate(features_testing, targets_testing)
print("Model 1 Test Accuracy: {:.4f}".format(test_accuracy1))

# Model 2 evaluation
test_loss2, test_accuracy2 = model2.evaluate(features_testing, targets_testing)
print("Model 2 Test Accuracy: {:.4f}".format(test_accuracy2))

# Predictions and Confusion Matrix

# Model 1 Predictions
targets_pred1 = model1.predict(features_testing)
targets_pred1 = numpy.argmax(targets_pred1, axis=1)

# Model 2 Predictions
targets_pred2 = model2.predict(features_testing)
targets_pred2 = numpy.argmax(targets_pred2, axis=1)

# Confusion Matrix for Model 1
conf_matrix1 = confusion_matrix(numpy.argmax(targets_testing, axis=1), targets_pred1)
f1_score1 = f1_score(numpy.argmax(targets_testing, axis=1), targets_pred1, average='weighted')
print("\nConfusion Matrix (Model 1):\n", conf_matrix1)
print("Average F1 Score (Model 1):", f1_score1)

# Confusion Matrix for Model 2
conf_matrix2 = confusion_matrix(numpy.argmax(targets_testing, axis=1), targets_pred2)
f1_score2 = f1_score(numpy.argmax(targets_testing, axis=1), targets_pred2, average='weighted')
print("\nConfusion Matrix (Model 2):\n", conf_matrix2)
print("Average F1 Score (Model 2):", f1_score2)
