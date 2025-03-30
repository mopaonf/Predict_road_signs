Traffic Sign Recognition Project

This project focuses on building a traffic sign recognition system using a 
convolutional neural network (CNN). The system includes a training script (`traffic.py`) 
and a GUI-based prediction tool (`predict_sign.py`).

To train the model, i used a dataset of traffic signs categorized into 43 classes. 
The model was trained with data augmentation and a learning rate scheduler to improve accuracy. 
For predictions, i developed a GUI that allows users to upload an image and view the predicted traffic sign.

The experimentation process involved multiple iterations of model tuning, 
including adjusting the number of layers, filters, and epochs. 
While the model performs well on most categories, 
i noticed challenges with certain signs due to similarities in appearance. 
Further improvements could involve collecting more diverse data and fine-tuning the model further.

run the predict_sign.py with the command "python predict_sign.py" in case you encounter errors try installing the required dependencies

opencv-python
scikit-learn
tensorflow
tkinter

with all these installed you will be good to go

also there are some categories with multiple images in the image folder this was just to better test the model