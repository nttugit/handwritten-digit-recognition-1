import os
import cv2 # computer vision
import numpy as np
import matplotlib.pyplot as plt # for visualization
import tensorflow as tf

# # 1. DATA PREPARATION

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# # print(x_train[0].shape)

# # PREPROCESSING
# # Chuẩn hoá từ 0-255 về 0-1
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
# # print(x_train[0])

# # 2. MODEL ARCHITECTURE

# # A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.
# model = tf.keras.models.Sequential()
# # add 1 flattened layer (from 28x29 to 1-D)
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# # Add a dense layer (a fully connected layer) with 128 neurons: for feature extraction
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# # 10 neurons for 10 classes
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# # adam optimizer adapts the learning rate for each parameter during training
# # sparse_categorical_crossentropy: calculates loss for multi-class classification (0, 1,2, ...)
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# # 3. TRAIN
# # train
# model.fit(x_train, y_train, epochs=3)

# # Save my model
# # model.save('handwritten.model')
# model.save('handwritten.h5')

model = tf.keras.models.load_model("handwritten.h5")

# 4. VALIDATION
# no validation here

# 5. EVALUATION
# loss, accuracy = model.evaluate(x_test, y_test)
# print(loss)
# print(accuracy)

# 6. INTERFERENCE - apply to the real-word use

image_number = 0
while os.path.isfile(f"digits/{image_number}_2.png"):
    try:
        img = cv2.imread(f"digits/{image_number}_2.png")[:,:,0]
        # pass image list as a numpy array
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        image_number += 1

# Dự đoán 9 số, sai hết 5
