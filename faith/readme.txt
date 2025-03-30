# Traffic Sign Recognition Project

This project is designed to train a neural network to recognize traffic signs and predict them using a graphical user interface (GUI).

## Prerequisites

1. **Python**: Ensure Python 3.7 or later is installed.
2. **Required Libraries**: Install the following Python libraries:
   - `tensorflow`
   - `numpy`
   - `opencv-python`
   - `Pillow`
   - `scikit-learn`
   - `tkinter` (comes pre-installed with Python on most systems)

   You can install the required libraries using:
   ```
   pip install tensorflow numpy opencv-python pillow scikit-learn
   ```

## Steps to Run the Project

### 1. Prepare the Dataset
- Place all traffic sign images in a directory.
- Ensure the filenames follow the format: `<category>_<description>.<extension>` (e.g., `0_stop_sign.png`).
- Example categories:
  ```
  0: Stop
  1: Speed Limit 50
  2: Speed Limit 20
  ...
  ```

### 2. Train the Model
- Run the `traffic.py` script to train the model:
  ```
  python traffic.py <data_directory> [model.h5]
  ```
  - Replace `<data_directory>` with the path to your dataset.
  - Optionally, specify a filename for the saved model (default is `best_model.h5`).

### 3. Test the GUI for Predictions
- Run the `predict_sign.py` script to launch the GUI:
  ```
  python predict_sign.py
  ```
- Use the GUI to:
  - Add an image of a traffic sign.
  - Predict the category and confidence of the traffic sign.

### 4. Model Details
- The model is a Convolutional Neural Network (CNN) with:
  - Three convolutional layers.
  - One dense hidden layer with dropout for regularization.
  - An output layer with softmax activation for classification.

### 5. Notes
- Ensure the dataset is balanced across all categories for optimal training.
- The GUI uses the trained model (`best_model.h5`) for predictions. Ensure this file exists in the project directory.

### 6. Example Commands
- Training:
  ```
  python traffic.py ./traffic_signs
  ```
- Running the GUI:
  ```
  python predict_sign.py
  ```

Enjoy using the Traffic Sign Recognition Project!
