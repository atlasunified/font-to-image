import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Layer
import pickle
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Preprocess the images
image_dir = 'images'
image_size = (512, 512)
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
images = np.empty((len(image_files), image_size[0], image_size[1], 3))
labels = []
le = LabelEncoder()

for i, image_file in enumerate(image_files):
    image = Image.open(image_file)
    image = image.resize(image_size)
    image = np.array(image) / 255.0
    images[i] = image

    label = os.path.basename(image_file).split('_')[0]
    labels.append(label)

labels = le.fit_transform(labels)
np.save('images.npy', images)
np.save('labels.npy', labels)

# Save the LabelEncoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# Load the images and labels
images = np.load('images.npy')
labels = np.load('labels.npy')

# Split the data into a training set and a test set
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Initialize the model
model = Sequential()

# Define GELU
class GELU(Layer):
    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

# Add layers
model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3)))
model.add(GELU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(GELU())
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3)))
model.add(GELU())
model.add(Flatten())
model.add(Dense(64))
model.add(GELU())
model.add(Dense(len(set(labels)), activation='softmax'))  # The output layer has one node per class, and uses softmax activation.

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=8, validation_data=(test_images, test_labels))

# Save the trained model
model.save('my_model.h5')

# Load the trained model
from tensorflow.keras.models import load_model
model = load_model('my_model.h5')

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

def predict_font(image_path, model, le):
    image_size = (512, 512)
    # Open and preprocess the image
    image = Image.open(image_path)
    image = image.resize(image_size)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Expand dimensions to represent a single 'batch' of one image

    # Use the model to make a prediction
    prediction = model.predict(image)

    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)

    # Decode the prediction
    predicted_font = le.inverse_transform([predicted_index])

    return predicted_font[0]

# Test the model
image_path = 'test.png'  # replace with your image path
predicted_font = predict_font(image_path, model, le)
print(f"The predicted font for the image {image_path} is {predicted_font}")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.tight_layout()
plt.show()