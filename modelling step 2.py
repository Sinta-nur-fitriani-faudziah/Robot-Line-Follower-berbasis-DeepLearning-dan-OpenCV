import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

# Fungsi untuk memuat dataset
def load_dataset(directory):
    X = []
    y = []
    for folder in os.listdir(directory):
        for file in os.listdir(os.path.join(directory, folder)):
            img_path = os.path.join(directory, folder, file)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (100, 100))  # Resize gambar menjadi 100x100 px
            X.append(img)
            y.append(folder)
    return np.array(X), np.array(y)

# Memuat dataset
X, y = load_dataset("dataset")

# Konversi label kategori menjadi angka
label_to_index = {label: i for i, label in enumerate(np.unique(y))}
y = np.array([label_to_index[label] for label in y])

# Membagi dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisasi data gambar
X_train = X_train / 255.0
X_test = X_test / 255.0

# Konversi label menjadi one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Membangun model CNN
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_to_index), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callback untuk menyimpan model terbaik
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Melatih model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint])

# Membuat grafik akurasi
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.savefig('accuracy_plot.png')
plt.show()

# Membuat grafik loss
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('loss_plot.png')
plt.show()

# Membuat confusion matrix
y_pred = np.argmax(model.predict(X_test), axis=-1)
conf_mat = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=label_to_index.keys(), yticklabels=label_to_index.keys())
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()
