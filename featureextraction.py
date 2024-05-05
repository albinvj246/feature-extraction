import numpy as np

def extract_features(motion_data):
    """
    Extract features from motion data.
    :param motion_data: A 3D array of shape (num_samples, 3) representing the motion data.
    :return: A 1D array of shape (num_features,) representing the extracted features.
    """
    # Calculate the mean and standard deviation of the motion data
    mean = np.mean(motion_data, axis=0)
    std = np.std(motion_data, axis=0)

    # Calculate the correlation between the x, y, and z axes
    correlation = np.corrcoef(motion_data.T)

    # Concatenate the mean, standard deviation, and correlation to form the feature vector
    features = np.concatenate([mean, std, correlation.flatten()])

    return features
