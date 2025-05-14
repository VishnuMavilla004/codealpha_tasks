
# Speech Emotion Recognition Using Deep Learning

import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.layers import TimeDistributed, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.callbacks import EarlyStopping
import librosa.display

# 1. Load and preprocess data
def extract_features(file_path, max_len=174):
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    pad_width = max_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_len]
    return mfccs

# 2. Load dataset (use RAVDESS or any speech emotion dataset)
data_path = "./RAVDESS"
emotions = []
features = []

emotion_dict = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

for root, _, files in os.walk(data_path):
    for file in files:
        if file.endswith(".wav"):
            emotion_label = emotion_dict[file.split("-")[2]]
            emotions.append(emotion_label)
            feature = extract_features(os.path.join(root, file))
            features.append(feature)

X = np.array(features)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Encode emotions
y = LabelEncoder().fit_transform(emotions)
y = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the model
model = Sequential()
model.add(Conv1D(64, kernel_size=5, strides=1, padding='same', activation='relu', input_shape=(40, 174)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64, return_sequences=False)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 4. Train the model
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[es])

# 5. Evaluate the model
score = model.evaluate(X_test, y_test)
print("Test accuracy:", score[1])

# 6. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')
plt.show()
