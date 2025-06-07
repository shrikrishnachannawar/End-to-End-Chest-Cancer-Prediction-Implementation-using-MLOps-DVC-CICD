import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import os


class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename

    def predict(self):
        ## load model
        #model = load_model(os.path.join("artifacts","training", "vgg16_chest_cancer.h5"), compile=False)
        model = load_model(os.path.join("model", "vgg16_chest_cancer.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = preprocess_input(test_image)

        prediction_score = model.predict(test_image)[0][0]  # Assuming binary classification with single output neuron

        threshold = 0.1
        label = 1 if prediction_score >= threshold else 0

        class_mapping = {
            0: 'malignant',
            1: 'normal'
        }

        predicted_label = class_mapping.get(label, 'Unknown')

        print(f"Predicted Label: {predicted_label}")

        return [{"predicted_label": predicted_label}]
