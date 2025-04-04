import tensorflow as tf
from keras import layers, models
import pandas as pd
import librosa
import librosa.util as util
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle

#################################################### Data Loading/Processing #################################################### 

# Load the CSV files
train_data = pd.read_csv('data/train_data.csv')
valid_data = pd.read_csv('data/valid_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Define which classes to keep/train for
allowed_actions = ['activate', 'deactivate', 'decrease', 'increase']
allowed_objects = ['lights', 'music', 'none', 'volume']

#Filter data
train_data = train_data[train_data['action'].isin(allowed_actions)]
train_data = train_data[train_data['object'].isin(allowed_objects)]

test_data = test_data[test_data['action'].isin(allowed_actions)]
test_data = test_data[test_data['object'].isin(allowed_objects)]

valid_data = valid_data[valid_data['action'].isin(allowed_actions)]
valid_data = valid_data[valid_data['object'].isin(allowed_objects)]

# Reset index after filtering
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)
valid_data.reset_index(drop=True, inplace=True)

# Create Label Encoders for each category: action, object, location
action_encoder = LabelEncoder()
object_encoder = LabelEncoder()
location_encoder = LabelEncoder()

#Fit encoders on training data
train_data['action'] = action_encoder.fit_transform(train_data['action'])
train_data['object'] = object_encoder.fit_transform(train_data['object'])
train_data['location'] = location_encoder.fit_transform(train_data['location'])

test_data['action'] = action_encoder.transform(test_data['action'])
test_data['object'] = object_encoder.transform(test_data['object'])
test_data['location'] = location_encoder.transform(test_data['location'])

valid_data['action'] = action_encoder.transform(valid_data['action'])
valid_data['object'] = object_encoder.transform(valid_data['object'])
valid_data['location'] = location_encoder.transform(valid_data['location'])

#Save encoders for later decoding
with open('encoders.pkl', 'wb') as f:
    pickle.dump((action_encoder, object_encoder, location_encoder), f)

# Example: Print unique encoded values
print("Action Classes:", action_encoder.classes_)
print("Object Classes:", object_encoder.classes_)
print("Location Classes:", location_encoder.classes_)

# Set display option to show all columns
pd.set_option('display.max_columns', None)

#Load/Preprocess the audio
def load_and_preprocess_audio(file_path, sr=16000, n_mfcc=13):
    # Load the audio file using librosa
    audio, _ = librosa.load(file_path, sr=sr)
    
    # Extract MFCC features (this time we get a sequence of MFCCs over time)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Optionally, you can use delta or delta-delta features for better recognition
    delta_mfcc = librosa.feature.delta(mfcc)
    delta_delta_mfcc = librosa.feature.delta(delta_mfcc)
    
    # Stack the MFCCs, delta, and delta-delta features (combine temporal features)
    features = np.vstack([mfcc, delta_mfcc, delta_delta_mfcc])
    
    # Flatten the features to get a single feature vector per audio file
    # You can either flatten or keep it in its temporal sequence for a recurrent model
    return features.T  # This will keep the temporal dimension

# Example: Load and preprocess one audio file
audio_path = train_data['path'].iloc[0]
mfcc_features = load_and_preprocess_audio(audio_path)

# Function to load and preprocess audio for all data
def preprocess_all_data(data, max_frames=300):
    features = []
    actions = []
    objects = []
    locations = []
    
    for idx, row in data.iterrows():
        audio_path = row['path']  # Assuming the 'path' column contains the path to the audio file
        
        # Extract features (MFCC, delta, delta-delta)
        audio_features = load_and_preprocess_audio(audio_path)
        
        # Pad or truncate audio features to a fixed size (max_frames x num_features)
        if audio_features.shape[0] < max_frames:
            # Pad with zeros if fewer frames than max_frames
            audio_features = util.fix_length(audio_features, size=max_frames, axis=0)
        else:
            # Truncate if more frames than max_frames
            audio_features = audio_features[:max_frames]
        
        features.append(audio_features)
        
        # Append the encoded labels for action, object, and location
        actions.append(row['action'])
        objects.append(row['object'])
        locations.append(row['location'])
    
    # Convert lists to numpy arrays
    features = np.array(features)
    actions = np.array(actions)
    objects = np.array(objects)
    locations = np.array(locations)
    
    return features, actions, objects, locations

# Preprocess the data (train, test, valid)
print("\nPreprocessing all data: ")
X_train, y_train_action, y_train_object, y_train_location = preprocess_all_data(train_data)
X_test, y_test_action, y_test_object, y_test_location = preprocess_all_data(test_data)
X_valid, y_valid_action, y_valid_object, y_valid_location = preprocess_all_data(valid_data)

# Print shapes to check
print("Features shape:", X_train.shape)
print("Action labels shape:", y_train_action.shape)
print("Object labels shape:", y_train_object.shape)
print("Location labels shape:", y_train_location.shape)
print("Data is ready for training.\n")








#################################################### Model Creation #################################################### 

def CNN_Model(input_shape, num_classes_action, num_classes_object, num_classes_location):
    model = models.Sequential()

    # 1D Convolutional Layer: 32 filters with kernel size 3, activation 'relu', and input shape as the feature dimension
    model.add(layers.Conv1D(32, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))  # Pooling layer to reduce dimensionality

    # Another 1D Convolutional Layer with 64 filters
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))

    # Another 1D Convolutional Layer with 128 filters
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.MaxPooling1D(2))

    # Flatten the output to feed into the dense layer
    model.add(layers.Flatten())

    # Fully connected layer for action classification
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))  # Dropout to prevent overfitting

    # Action classification output layer (Softmax for multi-class classification)
    model.add(layers.Dense(num_classes_action, activation='softmax', name='action_output'))

    # Object classification output layer (Softmax for multi-class classification)
    model.add(layers.Dense(num_classes_object, activation='softmax', name='object_output'))

    # Location classification output layer (Softmax for multi-class classification)
    model.add(layers.Dense(num_classes_location, activation='softmax', name='location_output'))

    # Compile the model
    model.compile(optimizer='adam',
                  loss={'action_output': 'sparse_categorical_crossentropy',
                        'object_output': 'sparse_categorical_crossentropy',
                        'location_output': 'sparse_categorical_crossentropy'},
                  metrics=['accuracy'])

    return model