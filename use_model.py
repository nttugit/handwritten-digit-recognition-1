from tensorflow.keras.models import load_model

# Load the saved model
my_handwritten_model = load_model("handwritten.h5")

# 4.

# 5. EVALUATION
loss, accuracy = my_handwritten_model.evaluate(x_test, y_test)

