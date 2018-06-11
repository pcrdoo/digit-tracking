from tensorflow.python import keras
from tensorflow.python.keras.models import model_from_json


class DigitClassifier():
    def __init__(self, img_rows, img_cols):
        self.num_classes = 10
        self.img_rows = img_rows
        self.img_cols = img_cols

        # Load and compile the model
        self.model = model_from_json(open('model/model.json', 'r').read())
        self.model.load_weights('model/weights.h5')
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

    def predict(self, imgs):
        # Return predictions
        imgs = imgs.reshape(imgs.shape[0], self.img_rows, self.img_cols, 1)
        return self.model.predict(imgs)
