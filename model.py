# model.py
import tensorflow as tf
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
    x = layers.Dropout(0.5)(x)

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
(X_train, y_train_a, y_train_o, y_train_l), (X_valid, y_valid_a, y_valid_o, y_valid_l), _ = get_data()

# Build and train model
input_shape = X_train.shape[1:]  # (frames, features)
model = CNN_Model(input_shape, num_classes_action=4, num_classes_object=3, num_classes_location=4)

print("Training model")
history = model.fit(X_train,
          {'action_output': y_train_a, 'object_output': y_train_o, 'location_output': y_train_l},
          validation_data=(X_valid,
                           {'action_output': y_valid_a, 'object_output': y_valid_o, 'location_output': y_valid_l}),
          epochs=20,
          batch_size=32,
          verbose=0)
print("Model trained. Model summary: ")
model.summary()
print("Saving model")
model.save("command_model.keras")
print("Model saved")

# Plot training and validation accuracy
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