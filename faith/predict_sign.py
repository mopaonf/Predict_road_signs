import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk

IMG_WIDTH = 30
IMG_HEIGHT = 30
CATEGORIES = {
    14: "Stop",
    37: "Speed Limit 50",
    0: "Speed Limit 20",
    17: "No Entry",
    13: "Yield",
    28: "Pedestrian Crossing",
    11: "Roundabout",
    7: "Traffic Light",
    13: "Turn Left",
    33: "Turn Right",
    23: "Slippery Road",
    5: "Slippery Road",
    1: "Speed Limit 40",
    10: "No Parking",
}

class TrafficSignPredictor:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Traffic Sign Predictor")
        self.model = tf.keras.models.load_model(model_path)
        self.image_path = None

        # Add Image Button
        self.add_image_button = Button(root, text="Add Image", command=self.add_image)
        self.add_image_button.grid(row=0, column=0, padx=10, pady=10)

        # Predict Button
        self.predict_button = Button(root, text="Predict", command=self.predict_sign)
        self.predict_button.grid(row=0, column=1, padx=10, pady=10)

        # Canvas to display the image
        self.canvas = Canvas(root, width=300, height=300, bg="gray")
        self.canvas.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        # Label to display the prediction
        self.result_label = Label(root, text="Prediction: ", font=("Arial", 16))
        self.result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    def add_image(self):
        """
        Open a file dialog to select an image and display it on the canvas.
        """
        self.image_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if self.image_path:
            image = Image.open(self.image_path)
            image = image.resize((300, 300))
            self.image_tk = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor="nw", image=self.image_tk)

    def predict_sign(self):
        """
        Predict the traffic sign in the uploaded image and display the result.
        """
        if not self.image_path:
            self.result_label.config(text="Prediction: No image selected")
            return

        # Get prediction
        prediction = predict_image(self.image_path, self.model)
        category, sign, probability = prediction

        # Display the result
        self.result_label.config(
            text=f"Prediction: {sign} (Category {category}, Confidence {probability:.2f})"
        )


def predict_image(image_path, model):
    """
    Predict the traffic sign for a given image path.

    Args:
        image_path (str): Path to the image.
        model (tf.keras.Model): Trained model.

    Returns:
        tuple: (category, sign, probability)
    """
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict the category
    prediction = model.predict(image, verbose=0)
    category = np.argmax(prediction)
    probability = np.max(prediction)
    sign = CATEGORIES.get(category, "Unknown")

    # Handle low-confidence predictions
    if probability < 0.5:
        sign = "Uncertain"

    return category, sign, probability


if __name__ == "__main__":
    # Initialize the GUI
    root = Tk()
    app = TrafficSignPredictor(root, "best_model.h5")
    root.mainloop()
