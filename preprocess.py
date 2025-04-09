# preprocess.py
import pandas as pd
import numpy as np
import librosa
import librosa.util as util
import pickle
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_audio(file_path, sr=16000, n_mfcc=13):
    audio, _ = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    features = np.vstack([mfcc, delta, delta2])
    return features.T

def preprocess_all_data(data, max_frames=300):
    features, actions, objects, locations = [], [], [], []
    for _, row in data.iterrows():
        audio_features = load_and_preprocess_audio(row['path'])
        if audio_features.shape[0] < max_frames:
            audio_features = util.fix_length(audio_features, size=max_frames, axis=0)
        else:
            audio_features = audio_features[:max_frames]
        features.append(audio_features)
        actions.append(row['action'])
        objects.append(row['object'])
        locations.append(row['location'])
    return np.array(features), np.array(actions), np.array(objects), np.array(locations)

def get_data():
    # Load CSVs
    train = pd.read_csv('data/train_data.csv')
    valid = pd.read_csv('data/valid_data.csv')
    test = pd.read_csv('data/test_data.csv')

    allowed_actions = ['activate', 'deactivate', 'decrease', 'increase']
    allowed_objects = ['lights', 'music', 'none', 'volume']

    # Filter
    for df in [train, valid, test]:
        df.drop(df[~df['action'].isin(allowed_actions)].index, inplace=True)
        df.drop(df[~df['object'].isin(allowed_objects)].index, inplace=True)
        df.reset_index(drop=True, inplace=True)

    # Encode
    action_encoder = LabelEncoder()
    object_encoder = LabelEncoder()
    location_encoder = LabelEncoder()

    train['action'] = action_encoder.fit_transform(train['action'])
    train['object'] = object_encoder.fit_transform(train['object'])
    train['location'] = location_encoder.fit_transform(train['location'])

    valid['action'] = action_encoder.transform(valid['action'])
    valid['object'] = object_encoder.transform(valid['object'])
    valid['location'] = location_encoder.transform(valid['location'])

    test['action'] = action_encoder.transform(test['action'])
    test['object'] = object_encoder.transform(test['object'])
    test['location'] = location_encoder.transform(test['location'])

    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump((action_encoder, object_encoder, location_encoder), f)

    # Preprocess audio
    X_train, y_train_a, y_train_o, y_train_l = preprocess_all_data(train)
    X_valid, y_valid_a, y_valid_o, y_valid_l = preprocess_all_data(valid)
    X_test, y_test_a, y_test_o, y_test_l = preprocess_all_data(test)

    return (X_train, y_train_a, y_train_o, y_train_l), \
           (X_valid, y_valid_a, y_valid_o, y_valid_l), \
           (X_test, y_test_a, y_test_o, y_test_l)
