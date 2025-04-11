import tensorflow as tf
import numpy as np
from preprocess import get_data  # Import the get_data function from preprocess.py

# Load the dataset (train, valid, test)
(X_train, _, _, _), (_, _, _, _), (_, _, _, _) = get_data()
print("Shape of X_train: ", X_train.shape)

# Convert the model to TensorFlow Lite format
model = tf.keras.models.load_model('command_model.keras')

# Enable quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_data_gen():
    # Use the first 100 samples of X_train for calibration
    for input_value in X_train[:100]:  # Adjust the number of samples if needed
        input_value = input_value.reshape(1, 300, 39)  # Reshape it to match the input shape
        yield [input_value.astype(np.float32)]  # Convert input to float32 if needed

converter.representative_dataset = representative_data_gen

# Convert the model
tflite_model = converter.convert()

# Save the quantized model to a file
with open('command_model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted and quantized successfully!")