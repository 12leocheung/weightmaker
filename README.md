Usage
1. Generating Training Data
The application requires JSON weights data to train the predictor. To generate data for specific input sizes (e.g., 2, 3, or 4 inputs):


Follow the prompts to enter the number of inputs and the number of samples. The files will be saved as weights_<N>inputs.json.

2. Running the Application
Start the main desktop application:

3. Analysing Code
Navigate to the Code Analyser tab in the app.

Click Upload Neural Network (.py).

Select small_example.py or large_example.py to see the analyser detect the network dimensions and targets, and instantly predict the optimal initial weights!

Technical Details
UI Framework: PyQt6

Deep Learning: PyTorch (torch.nn)

Parsing: ast module (Static analysis of uploaded .py files safely without executing them)

Math/Matrix Generation: NumPy
