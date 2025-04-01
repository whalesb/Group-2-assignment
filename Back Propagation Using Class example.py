# Import required libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data handling

# ===== DATA LOADING AND PREPROCESSING =====
# Load the dataset from CSV file
data = pd.read_csv("C:/Users/Wale/Documents/tetsting_back_propagation.csv")

# Extract input features (Science, Maths, Hours) and convert to numpy array
X = data[['Science', 'Maths', 'Hours']].values

# Extract target variable (Cpe_593), reshape to column vector
y = data['Cpe_593'].values.reshape(-1, 1)

# Normalize all values to range [0,1] by dividing by 100
X = X / 100  # Normalize input features (assuming scores out of 100)
y = y / 100  # Normalize target values

# ===== NEURAL NETWORK ARCHITECTURE =====
# Define network architecture sizes
input_size = 3    # 3 input features (Science, Maths, Hours)
hidden_size = 2   # 2 neurons in hidden layer
output_size = 1   # 1 output neuron (Cpe_593)

# ===== WEIGHT INITIALIZATION =====
# Set random seed for reproducibility
np.random.seed(42)

# Initialize weights:
# W1: weights from input to hidden layer (3 inputs × 2 hidden neurons)
W1 = np.random.randn(input_size, hidden_size)

# W2: weights from hidden to output layer (2 hidden × 1 output)
W2 = np.random.randn(hidden_size, output_size)

# Initialize biases:
# b1: biases for hidden layer (1 × 2)
b1 = np.zeros((1, hidden_size))

# b2: bias for output layer (1 × 1)
b2 = np.zeros((1, output_size))

# ===== TRAINING PARAMETERS =====
learning_rate = 0.1  # Step size for weight updates
epochs = 10000       # Number of training iterations

# ===== ACTIVATION FUNCTIONS =====
def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid function (for backpropagation)"""
    return x * (1 - x)

# ===== TRAINING LOOP =====
for epoch in range(epochs):
    # ----- FORWARD PASS -----
    # Calculate hidden layer input and apply activation
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Calculate output layer input and apply activation
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    y_pred = sigmoid(output_layer_input)  # Network predictions
    
    # ----- ERROR CALCULATION -----
    error = y - y_pred  # Difference between actual and predicted
    
    # ----- BACKPROPAGATION -----
    # Calculate gradient at output layer
    d_predicted_output = error * sigmoid_derivative(y_pred)
    
    # Propagate error back to hidden layer
    error_hidden_layer = d_predicted_output.dot(W2.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # ----- WEIGHT UPDATES -----
    # Update hidden-to-output weights and bias
    W2 += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    # Update input-to-hidden weights and biases
    W1 += X.T.dot(d_hidden_layer) * learning_rate
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
    
    # Print progress every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(np.square(error))  # Mean Squared Error
        print(f"Epoch {epoch}, Loss: {loss:.6f}")

# ===== POST-TRAINING EVALUATION =====
# Make final predictions with trained weights
hidden_layer_input = np.dot(X, W1) + b1
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, W2) + b2
y_pred = sigmoid(output_layer_input)

# Denormalize values back to original scale (0-100)
y_pred_denorm = y_pred * 100  # Predicted scores
y_actual_denorm = y * 100     # Actual scores
X_denorm = X * 100           # Input features

# Calculate squared errors for each sample
squared_errors = (y_actual_denorm.flatten() - y_pred_denorm.flatten()) ** 2

# ===== RESULTS PRESENTATION =====
# Create results table
results = pd.DataFrame({
    'Science': X_denorm[:, 0],
    'Maths': X_denorm[:, 1],
    'Hours': X_denorm[:, 2],
    'Actual Cpe_593': y_actual_denorm.flatten(),
    'Predicted Cpe_593': y_pred_denorm.flatten(),
    'Squared Error': squared_errors
})

# Print prediction results
print("\nPrediction Results:")
print(results.to_string(index=False))

# Print learned parameters in matrix form
print("\nFinal Weights and Biases (Matrix Form):")
print("W1 (input to hidden):\n", W1)
print("b1 (hidden bias):\n", b1)
print("W2 (hidden to output):\n", W2)
print("b2 (output bias):\n", b2)

# Print individual weights and biases with clear labels
print("\nIndividual Weights and Biases:")
print(f"w1 (Science → Hidden Neuron 1): {W1[0, 0]:.6f}")
print(f"w2 (Maths → Hidden Neuron 1): {W1[1, 0]:.6f}")
print(f"w3 (Hours → Hidden Neuron 1): {W1[2, 0]:.6f}")
print(f"w4 (Science → Hidden Neuron 2): {W1[0, 1]:.6f}")
print(f"w5 (Maths → Hidden Neuron 2): {W1[1, 1]:.6f}")
print(f"w6 (Hours → Hidden Neuron 2): {W1[2, 1]:.6f}")
print(f"w7 (Hidden Neuron 1 → Output): {W2[0, 0]:.6f}")
print(f"w8 (Hidden Neuron 2 → Output): {W2[1, 0]:.6f}")
print(f"b1 (Bias for Hidden Neuron 1): {b1[0, 0]:.6f}")
print(f"b2 (Bias for Hidden Neuron 2): {b1[0, 1]:.6f}")
print(f"b3 (Bias for Output Neuron): {b2[0, 0]:.6f}")

# Calculate and print final Mean Squared Error
mse = np.mean(squared_errors)
print(f"\nMean Squared Error: {mse:.2f}")