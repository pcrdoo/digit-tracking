from __future__ import print_function

from tensorflow.python import keras
from tensorflow.python.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                            MaxPooling2D)
from tensorflow.python.keras.models import Sequential

from dataset import MnistDataset

# Create the dataset
dataset = MnistDataset(augmentation='pad')

# Params
batch_size = 128
epochs = 12

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=dataset.shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(dataset.num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(dataset.x_train, dataset.y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(dataset.x_test, dataset.y_test))

# Evaluate
score = model.evaluate(dataset.x_test, dataset.y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Save
model_json = model.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model/weights.h5')
