import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

xor_data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'Output': [0, 1, 1, 0]
})

xor_model = keras.Sequential([
    keras.layers.Input(shape=(2,)),  
    keras.layers.Dense(4, activation='relu'),  
    keras.layers.Dense(1, activation='sigmoid')  
])

optimizer = keras.optimizers.Adam(learning_rate=0.01)
xor_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

X_xor = xor_data[['A', 'B']].values
y_xor = xor_data['Output'].values
xor_model.fit(X_xor, y_xor, epochs=5000, verbose=0)  # Increased epochs

xor_model.save('xor_model.h5')

# Acc
test_data_xor = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'Output': [0, 1, 1, 0]  # Expected outputs for XOR
})

X_test_xor = test_data_xor[['A', 'B']].values
y_test_xor = test_data_xor['Output'].values

# Evaluate the XOR model on the test data
loss, accuracy_xor = xor_model.evaluate(X_test_xor, y_test_xor, verbose=0)
print(f"XOR Gate Accuracy: {accuracy_xor * 100}%")
