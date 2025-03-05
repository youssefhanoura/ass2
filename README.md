# ass2

import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

np.random.seed(42)
w1 = np.random.uniform(-0.5, 0.5, (2, 3))  # 2 inputs, 3 hidden neurons
w2 = np.random.uniform(-0.5, 0.5, (3, 1))  # 3 hidden neurons, 1 output neuron
b1, b2 = 0.5, 0.7  # Biases


X = np.array([[0.1, 0.2]])  # Example input
Y = np.array([[0.3]])  # Target output

# Forward pass
hidden_input = np.dot(X, w1) + b1
hidden_output = tanh(hidden_input)
final_input = np.dot(hidden_output, w2) + b2
final_output = tanh(final_input)
print("Output before training:", final_output)

# Backpropagation
learning_rate = 0.1
epochs = 1000

for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, w1) + b1
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, w2) + b2
    final_output = tanh(final_input)
    
    # Compute error
    error = Y - final_output
    
    # Backpropagation
    d_final = error * tanh_derivative(final_output)
    d_hidden = d_final.dot(w2.T) * tanh_derivative(hidden_output)
    
    # Update weights and biases
    w2 += learning_rate * hidden_output.T.dot(d_final)
    w1 += learning_rate * X.T.dot(d_hidden)
    b2 += learning_rate * np.sum(d_final)
    b1 += learning_rate * np.sum(d_hidden)

print("Output after training:", final_output)
import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2 

np.random.seed(42)
w1 = np.random.uniform(-0.5, 0.5, (2, 3))  


X = np.array([[0.1, 0.2]])  
Y = np.array([[0.3]])  

# Forward pass (before training)
hidden_input = np.dot(X, w1) + b1  # Compute input to hidden layer
hidden_output = tanh(hidden_input)  # Apply activation function
final_input = np.dot(hidden_output, w2) + b2  # Compute input to output neuron
final_output = tanh(final_input)  # Apply activation function
print("Forward output before training:", final_output)

# Backpropagation parameters
learning_rate = 0.1
epochs = 1000

# Training loop
for _ in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, w1) + b1  # Compute hidden layer input
    hidden_output = tanh(hidden_input)  # Apply tanh activation
    final_input = np.dot(hidden_output, w2) + b2  # Compute output layer input
    final_output = tanh(final_input)  # Apply tanh activation
    

    error = Y - final_output  # Difference between actual and predicted output
    
    # Backpropagation
    d_final = error * tanh_derivative(final_output)  # Gradient for output layer
    d_hidden = d_final.dot(w2.T) * tanh_derivative(hidden_output)  # Gradient for hidden layer
    
    # Update weights and biases
    w2 += learning_rate * hidden_output.T.dot(d_final)  # Update weights from hidden to output
    w1 += learning_rate * X.T.dot(d_hidden)  # Update weights from input to hidden
    b2 += learning_rate * np.sum(d_final)  # Update bias for output layer
    b1 += learning_rate * np.sum(d_hidden)  # Update bias for hidden layer

    # Print output during training at certain intervals
    if _ % 200 == 0:
        print(f"Epoch {_}: Backpropagation output =", final_output)

# Output after training
print("Forward output after training:", final_output)



Forward output before training: [[0.30240427]]
Epoch 0: Backpropagation output = [[0.30240427]]
Epoch 200: Backpropagation output = [[0.3]]
Epoch 400: Backpropagation output = [[0.3]]
Epoch 600: Backpropagation output = [[0.3]]
Epoch 800: Backpropagation output = [[0.3]]
Forward output after training: [[0.3]]
