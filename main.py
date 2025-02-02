import streamlit as st
import numpy as np
import joblib
from tensorflow.keras import models

class Perceptron:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = np.random.rand(3)  # Two inputs + bias

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

# Load the NAND model
nand_model = joblib.load('nand_perceptron.pkl')

# Load the XOR model
xor_model = models.load_model('xor_model.h5')

# Streamlit App
st.title("Logic Gates using Neural Networks")

# Sidebar for choosing logic gate
st.sidebar.title("Select Logic Gate")
option = st.sidebar.selectbox("Choose a gate", ("NAND", "XOR"))

if option == "NAND":
    st.header("NAND Gate")
    a_nand = st.selectbox("Input A", [0, 1])
    b_nand = st.selectbox("Input B", [0, 1])
    if st.button("Predict NAND"):
        result_nand = nand_model.predict(np.array([a_nand, b_nand]))
        st.write(f"NAND({a_nand}, {b_nand}) = {result_nand}")

elif option == "XOR":
    st.header("XOR Gate")
    a_xor = st.selectbox("Input A ", [0, 1], key="xor_a")
    b_xor = st.selectbox("Input B ", [0, 1], key="xor_b")
    if st.button("Predict XOR"):
        result_xor = xor_model.predict(np.array([[a_xor, b_xor]]))
        # Convert the decimal output to integer
        st.write(f"XOR({a_xor}, {b_xor}) = {int(np.round(result_xor[0][0]))}")
