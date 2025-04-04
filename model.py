import tensorflow as tf
import pandas as pd
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle


# Load the CSV files
train_data = pd.read_csv('data/train_data.csv')
valid_data = pd.read_csv('data/valid_data.csv')
test_data = pd.read_csv('data/test_data.csv')

# Create Label Encoders for each category: action, object, location
action_encoder = LabelEncoder()
object_encoder = LabelEncoder()
location_encoder = LabelEncoder()

#Fit encoders on training data
train_data['action'] = action_encoder.fit_transform(train_data['action'])
train_data['object'] = object_encoder.fit_transform(train_data['object'])
train_data['location'] = location_encoder.fit_transform(train_data['location'])

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

print(mfcc_features.shape)  # Should be (frames, 39), where 39 = 13 MFCCs + 13 delta + 13 delta-delta



