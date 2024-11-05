import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def apply_projection_to_dict(input_dict, n_components=2, perplexity=10, random_state=42, proj='tsne', metric="euclidean"):
    """takes an input_dict
        {person:{label:tensor_of_predictions}}
        tensor of predictions is of shape (batch_size,smth)
        and takes a projection of all predictions and gets a new dictionary with
        {person:{label:projected_predictions}}"""
    # Collect all predictions into a single array
    all_predictions = []
    for person, label_dict in input_dict.items():
        for label, predictions in label_dict.items():
            all_predictions.append(predictions.numpy())

    # Concatenate all predictions
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Apply t-SNE
    if proj == 'tsne':
        projection = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, metric=metric)
    elif proj == 'pca':
        projection = PCA(n_components=n_components)
    # tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    if proj == 'none':
        projected_predictions = all_predictions[:,:n_components]
    else:
        projected_predictions = projection.fit_transform(all_predictions)

    # Create new dictionary with projected predictions
    new_dict = {}
    start_idx = 0
    for person, label_dict in input_dict.items():
        new_dict[person] = {}
        for label, predictions in label_dict.items():
            end_idx = start_idx + predictions.shape[0]
            new_dict[person][label] = tf.convert_to_tensor(projected_predictions[start_idx:end_idx], dtype=tf.float32)
            start_idx = end_idx

    return new_dict


def filter_dict_by_keys(original_dict, keys_list):
    new_dict = {}

    for key in keys_list:
        if key in original_dict:
            new_dict[key] = original_dict[key]
        else:
            print(f"Key '{key}' not found in dictionary")

    return new_dict