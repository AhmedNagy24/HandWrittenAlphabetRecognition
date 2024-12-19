import numpy
import pandas
import sklearn
import tensorflow
from PIL import Image
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC, LinearSVC
from tensorflow.keras.layers import Dense

# Data exploration and preparation
# Data loading
data = pandas.read_csv('/content/drive/MyDrive/A_Z Handwritten Data.csv')

# Identify the number of unique classes and show their distribution
print("Number of unique values:", data.iloc[:, 0].nunique(), "\n")
print("Data distribution: \n", data.iloc[:, 0].value_counts())

# Normalize each image
images = data.iloc[:, 1:].values
images = images / 255.0

# Reshape the flattened vectors to reconstruct and display the corresponding images while testing the models.
features = images.reshape(len(images), 28, 28)

# Split the data into training and testing datasets
targets = data.iloc[:, 0]
features, targets = sklearn.utils.shuffle(features, targets)

size_of_training = int(len(data) * 0.8)

features_training = features[:size_of_training]
features_testing = features[size_of_training:]

targets_training = targets[:size_of_training]
targets_testing = targets[size_of_training:]

# First experiment (You can use scikit-learn):
# Train SVM with a linear kernel
svm_linear = LinearSVC()
svm_linear.fit(features_training.reshape(len(features_training), 784), targets_training)
targets_pred_linear = svm_linear.predict(features_testing.reshape(len(features_testing), 784))
conf_matrix_linear = confusion_matrix(targets_testing, targets_pred_linear)
f1_score_linear = f1_score(targets_testing, targets_pred_linear, average='weighted')
print("Confusion Matrix (Linear Kernel):\n", conf_matrix_linear)
print("Average F1 Score (Linear Kernel):", f1_score_linear, "\n")

# Train SVM with a non-linear kernel
svm_non_linear = SVC(kernel='rbf')
svm_non_linear.fit(features_training.reshape(len(features_training), 784), targets_training)
targets_pred_non_linear = svm_non_linear.predict(features_testing.reshape(len(features_testing), 784))
conf_matrix_non_linear = confusion_matrix(targets_testing, targets_pred_non_linear)
f1_score_non_linear = f1_score(targets_testing, targets_pred_non_linear, average='weighted')
print("Confusion Matrix (Non Linear Kernel):\n", conf_matrix_non_linear)
print("Average F1 Score (Non Linear Kernel):", f1_score_non_linear)

# Split the training dataset into training and validation datasets.
size_of_validation = int(len(features_training) * 0.9)

features_train = features_training[:size_of_validation]
features_valid = features_training[size_of_validation:]

targets_train = targets_training[:size_of_validation]
targets_valid = targets_training[size_of_validation:]

# Third experiment (You can use TensorFlow):
# Design 2 Neural Networks (with different number of hidden layers, neurons, activations, etc.)
# First Neural Network: Simple architecture with one hidden layer
model1 = tensorflow.keras.models.Sequential()
model1.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
model1.add(Dense(128, activation='relu'))
model1.add(Dense(26, activation='softmax'))

model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Second Neural Network: More complex with two hidden layers
model2 = tensorflow.keras.models.Sequential()
model2.add(tensorflow.keras.layers.Flatten(input_shape=(28, 28)))
model2.add(Dense(256, activation='relu'))
model2.add(Dense(128, activation='relu'))
model2.add(Dense(26, activation='softmax'))

model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train each one of these models and plot the error and accuracy curves for the training data and validation datasets.
history1 = model1.fit(features_train, targets_train, epochs=4, validation_data=(features_valid, targets_valid))
history2 = model2.fit(features_train, targets_train, epochs=4, validation_data=(features_valid, targets_valid))


def plot_history(history, model_name):
    pyplot.figure(figsize=(12, 5))

    pyplot.subplot(1, 2, 1)
    pyplot.plot(history.history['accuracy'], label='Train Accuracy')
    pyplot.plot(history.history['val_accuracy'], label='Validation Accuracy')
    pyplot.title(f'{model_name} Accuracy')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Accuracy')
    pyplot.legend()

    pyplot.subplot(1, 2, 2)
    pyplot.plot(history.history['loss'], label='Train error')
    pyplot.plot(history.history['val_loss'], label='Validation error')
    pyplot.title(f'{model_name} error')
    pyplot.xlabel('Epochs')
    pyplot.ylabel('Error')
    pyplot.legend()

    pyplot.show()


plot_history(history1, 'Simple Model')
plot_history(history2, 'Complex Model')

# Save the best model in a separated file, then reload it.
_, valid_acc_1 = model1.evaluate(features_valid, targets_valid)
_, valid_acc_2 = model2.evaluate(features_valid, targets_valid)

if valid_acc_1 > valid_acc_2:
    best_model = model1
    best_model.save('best_model.keras')
else:
    best_model = model2
    best_model.save('best_model.keras')

reloaded_model = tensorflow.keras.models.load_model('best_model.h5')

# Test the best model and provide the confusion matrix and the average f-1 scores for the testing data.
test_loss, test_accuracy = reloaded_model.evaluate(features_testing, targets_testing)
print("The best model Test Accuracy: {:.4f}".format(test_accuracy))

targets_pred = reloaded_model.predict(features_testing)
targets_pred = numpy.argmax(targets_pred, axis=1)

conf_matrix = confusion_matrix(targets_testing, targets_pred)
f1_score = f1_score(targets_testing, targets_pred, average='weighted')
print("\nConfusion Matrix (The best model):\n", conf_matrix)
print("Average F1 Score (The best model):", f1_score)


def show_image(path):
    img = tensorflow.keras.preprocessing.image.load_img(path, color_mode='grayscale')
    pyplot.imshow(img, cmap='gray')
    pyplot.axis('off')
    pyplot.show()


image_paths = ["/content/drive/MyDrive/TestImages/A.png", "/content/drive/MyDrive/TestImages/D.png",
               "/content/drive/MyDrive/TestImages/E.png",
               "/content/drive/MyDrive/TestImages/F.png", "/content/drive/MyDrive/TestImages/H.png",
               "/content/drive/MyDrive/TestImages/I.png",
               "/content/drive/MyDrive/TestImages/K.png", "/content/drive/MyDrive/TestImages/L.png",
               "/content/drive/MyDrive/TestImages/M.png",
               "/content/drive/MyDrive/TestImages/O.png", "/content/drive/MyDrive/TestImages/R.png",
               "/content/drive/MyDrive/TestImages/S.png",
               "/content/drive/MyDrive/TestImages/Z.png"]

get_letter = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}
test_images = []
for img_path in image_paths:
    try:
        img = Image.open(img_path)
        img = img.resize((28, 28))
        img = img.convert('L')
        img_array = numpy.array(img) / 255.0
        test_images.append(img_array)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

for i in range(len(test_images)):
    predict = reloaded_model.predict(test_images[i].reshape(1, 28, 28))
    predict = numpy.argmax(predict, axis=1)
    print(f"Image {i + 1}")
    show_image(image_paths[i])
    print(f"Prediction {get_letter[predict[0]]}")
