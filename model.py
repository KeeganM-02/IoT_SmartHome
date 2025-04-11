# model.py
import tensorflow as tf
import numpy as np
from keras import layers, models
from preprocess import get_data
import matplotlib.pyplot as plt

def CNN_Model(input_shape, num_classes_action, num_classes_object, num_classes_location):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, 3, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 3, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.7)(x)

    # Separate output layers for action, object, location
    output_action = layers.Dense(num_classes_action, activation='softmax', name='action_output')(x)
    output_object = layers.Dense(num_classes_object, activation='softmax', name='object_output')(x)
    output_location = layers.Dense(num_classes_location, activation='softmax', name='location_output')(x)

    model = models.Model(inputs=inputs, outputs=[output_action, output_object, output_location])

    model.compile(optimizer='adam',
                  loss={
                      'action_output': 'sparse_categorical_crossentropy',
                      'object_output': 'sparse_categorical_crossentropy',
                      'location_output': 'sparse_categorical_crossentropy'
                  },
                  metrics={'action_output': 'accuracy',
                      'object_output': 'accuracy',
                      'location_output': 'accuracy'})

    return model

# Load preprocessed data
(X_train, y_train_a, y_train_o, y_train_l), (X_valid, y_valid_a, y_valid_o, y_valid_l), (X_test, y_test_a, y_test_o, y_test_l) = get_data()

# Build and train model
input_shape = X_train.shape[1:]  # (frames, features)
model = CNN_Model(input_shape, num_classes_action=4, num_classes_object=3, num_classes_location=4)

#Train model
print("Training model")
history = model.fit(X_train,
          {'action_output': y_train_a, 'object_output': y_train_o, 'location_output': y_train_l},
          validation_data=(X_valid,
                           {'action_output': y_valid_a, 'object_output': y_valid_o, 'location_output': y_valid_l}),
          epochs=20,
          batch_size=32,
          verbose=0)


print("Model summary: ")
model.summary()
print("Evaluating model on test data set")

print("Saving model")
model.save("command_model.keras")
print("Model saved")

# Plot training, validation, and test accuracy
plt.figure(figsize=(12, 6))

# Action accuracy
plt.subplot(1, 3, 1)
plt.plot(history.history['action_output_accuracy'], label='train action accuracy')
plt.plot(history.history['val_action_output_accuracy'], label='val action accuracy')
plt.title('Action Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Object accuracy
plt.subplot(1, 3, 2)
plt.plot(history.history['object_output_accuracy'], label='train object accuracy')
plt.plot(history.history['val_object_output_accuracy'], label='val object accuracy')
plt.title('Object Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Location accuracy
plt.subplot(1, 3, 3)
plt.plot(history.history['location_output_accuracy'], label='train location accuracy')
plt.plot(history.history['val_location_output_accuracy'], label='val location accuracy')
plt.title('Location Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Select a random index for a test example
index = np.random.randint(0, len(X_test))

# Extract the features and labels for this example
sample_features = X_test[index]  # This is the MFCC features of the sample
true_action = y_test_a[index]
true_object = y_test_o[index]
true_location = y_test_l[index]

# Reshape if needed (assuming your model expects input with shape (frames, features))
sample_features = np.expand_dims(sample_features, axis=0)  # Add batch dimension

# Predict using the model
predictions = model.predict(sample_features)

# Get the predicted action, object, and location
predicted_action = np.argmax(predictions[0])
predicted_object = np.argmax(predictions[1])
predicted_location = np.argmax(predictions[2])

# Print the results
print(f"True Action: {true_action}, Predicted Action: {predicted_action}")
print(f"True Object: {true_object}, Predicted Object: {predicted_object}")
print(f"True Location: {true_location}, Predicted Location: {predicted_location}")