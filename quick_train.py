import numpy as np, os
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

print("Loading data...")
X = np.load('model/X.npy')
y = np.load('model/y.npy')
labels = np.load('model/labels.npy', allow_pickle=True)
print(f"X={X.shape}, Classes={list(labels)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Flatten sequences — no LSTM, just Dense layers (much faster)
X_train_flat = X_train.reshape(len(X_train), -1)
X_test_flat  = X_test.reshape(len(X_test), -1)
print(f"Flattened: {X_train_flat.shape}")

print("Building model...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train_flat.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dense(len(labels), activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nTraining... should be fast now!")
model.fit(
    X_train_flat, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_flat, y_test),
    verbose=1,
    callbacks=[
        ModelCheckpoint('model/sign_language_model.h5', save_best_only=True, monitor='val_accuracy', verbose=0),
        EarlyStopping(patience=20, restore_best_weights=True, verbose=1)
    ]
)

loss, acc = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {acc*100:.2f}%")
np.save('model/labels.npy', labels)
# Save flatten shape for inference
np.save('model/input_shape.npy', np.array(X_train_flat.shape[1]))
print("Model saved!")
