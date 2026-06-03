import numpy as np

# 1. Activation Function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Input dataset (4 samples, 2 features each)
# Let's teach it the XOR gate logic
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target outputs (Actual answers)
y = np.array([[0], [1], [1], [0]])

# 3. Initialize Weights and Biases randomly
np.random.seed(42) # For reproducible results

# Layer 1 (Weights connecting 2 inputs to 3 hidden neurons)
weights_input_hidden = np.random.uniform(size=(2, 3))
bias_hidden = np.random.uniform(size=(1, 3))

# Layer 2 (Weights connecting 3 hidden neurons to 1 output neuron)
weights_hidden_output = np.random.uniform(size=(3, 1))
bias_output = np.random.uniform(size=(1, 1))

learning_rate = 0.5

# 4. Training Loop
for epoch in range(10000):
    # --- FORWARD PROPAGATION ---
    # Hidden layer activation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    # Output layer activation
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)
    
    # --- BACKPROPAGATION ---
    # Calculate error at output
    error = y - predicted_output
    
    # Calculate gradients (how much we need to change weights)
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    
    # Calculate error at hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)
    
    # --- UPDATING WEIGHTS AND BIASES ---
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    
    weights_input_hidden += X.T.dot(d_hidden_layer) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# 5. Test the trained model
print("Predicted Outputs after 10,000 epochs:")
print(np.round(predicted_output, 3))