import numpy as np
import pandas as pd
import joblib

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(3) 

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation(summation)

    def train(self, training_inputs, labels, epochs):
        for _ in range(epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * error * inputs
                self.weights[0] += self.learning_rate * error

# Create dataset for NAND gate
nand_data = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'Output': [1, 1, 1, 0]
})

perceptron = Perceptron()
X_nand = nand_data[['A', 'B']].values
y_nand = nand_data['Output'].values
perceptron.train(X_nand, y_nand, epochs=100)

# Save the model
joblib.dump(perceptron, 'nand_perceptron.pkl')

test_data_nand = pd.DataFrame({
    'A': [0, 0, 1, 1],
    'B': [0, 1, 0, 1],
    'Output': [1, 1, 1, 0]  # Expected outputs for NAND
})

X_test_nand = test_data_nand[['A', 'B']].values
y_test_nand = test_data_nand['Output'].values

predictions_nand = [perceptron.predict(inputs) for inputs in X_test_nand]

# Calculate accuracy
accuracy_nand = np.mean(np.array(predictions_nand) == y_test_nand) * 100
print(f"NAND Gate Accuracy: {accuracy_nand}%")
