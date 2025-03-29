import cv2
import numpy as np
import tensorflow as tf
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk
from traffic import IMG_WIDTH, IMG_HEIGHT

class TrafficSignRecognizer:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Traffic Sign Recognizer")
        self.model = tf.keras.models.load_model(model_path)
        self.image_path = None
        self.image_canvas = None

        # Add Image Button
        self.add_image_button = Button(root, text="Add Image", command=self.add_image)
        self.add_image_button.grid(row=0, column=0, padx=10, pady=10)

        # Results Button
        self.results_button = Button(root, text="Show Results", command=self.show_results)
        self.results_button.grid(row=0, column=1, padx=10, pady=10)

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

    def show_results(self):
        """
        Predict the category of the uploaded image and display the result.
        """
        if not self.image_path:
            self.result_label.config(text="Prediction: No image selected")
            return

        # Load and preprocess the image
        image = cv2.imread(self.image_path)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Predict the category
        prediction = self.model.predict(image, verbose=0)
        predicted_category = np.argmax(prediction)

        # Display the result
        self.result_label.config(text=f"Prediction: {predicted_category}")


if __name__ == "__main__":
    # Initialize the GUI
    root = Tk()
    app = TrafficSignRecognizer(root, "traffic_model.h5")
    root.mainloop()
