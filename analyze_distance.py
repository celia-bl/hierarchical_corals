from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from img_transformation import *
from sklearn.decomposition import PCA


def calculate_distances(target_vectors, target_img_names, center):
    distances_to_center = cdist(target_vectors, [center], metric='euclidean').flatten()
    pairwise_distances = cdist(target_vectors, target_vectors, metric='euclidean')
    avg_pairwise_distance = np.mean(pairwise_distances[np.triu_indices_from(pairwise_distances, k=1)])

    closest_to_center_index = np.argmin(distances_to_center)
    closest_to_center_img_name = target_img_names[closest_to_center_index]

    farthest_to_center_index = np.argmax(distances_to_center)
    farthest_to_center_img_name = target_img_names[farthest_to_center_index]

    farthest_indices = np.argsort(distances_to_center)[-4:]
    farthest_img_names = [target_img_names[idx] for idx in farthest_indices]

    max_pairwise_distance_indices = np.unravel_index(np.argmax(pairwise_distances), pairwise_distances.shape)
    most_distant_img_names = [target_img_names[idx] for idx in max_pairwise_distance_indices]

    return {
        'center': center,
        'avg_pairwise_distance': avg_pairwise_distance,
        'closest_to_center_img_name': closest_to_center_img_name,
        'farthest_to_center_img_name': farthest_to_center_img_name,
        'farthest_img_names': farthest_img_names,
        'most_distant_img_names': most_distant_img_names,
        'distances_to_center': distances_to_center[farthest_indices]
    }


def calculate_distance_features(img_names, features, labels, target_label):
    target_indices = [i for i in range(len(labels)) if labels[i] == target_label]
    target_vectors = np.array([features[i] for i in target_indices])
    target_img_names = [img_names[i] for i in target_indices]
    center = np.mean(target_vectors, axis=0)

    print(center)
    normal_dict = calculate_distances(target_vectors, target_img_names, center)

    pca = PCA(n_components=2)
    pca.fit(target_vectors)
    transformed_vectors = pca.transform(target_vectors)

    center_pca = [0, 0]
    print(center_pca)
    pca_dict = calculate_distances(transformed_vectors, target_img_names, center_pca)

    plt.figure(figsize=(8, 6))
    plt.scatter(transformed_vectors[:, 0], transformed_vectors[:, 1], c='b', marker='o', label='PCA Components')
    plt.scatter(0, 0, c='r', marker='x', label='Center')
    plt.title(f'Scatter plot of PCA components for label {target_label}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.show()

    return normal_dict, pca_dict


def plot_distant_features(dict_distance, annotations_path, img_folder, target_label, name):
    closest_img_name = dict_distance['closest_to_center_img_name']
    farthest_img_names = dict_distance['farthest_img_names']
    avg_pairwise_distance = dict_distance['avg_pairwise_distance']
    distances_to_center = dict_distance['distances_to_center']

    fig, axs = plt.subplots(1, 5, figsize=(20, 5))

    closest_img = get_patch_image_pt(closest_img_name, annotations_path, img_folder)
    axs[0].imshow(closest_img)
    axs[0].set_title('Closest to Center')

    for i, (farthest_img_name, distance) in enumerate(zip(farthest_img_names, distances_to_center)):
        farthest_img = get_patch_image_pt(farthest_img_name, annotations_path, img_folder)
        axs[i + 1].imshow(farthest_img)
        axs[i + 1].set_title(f'Far {i + 1}: {distance:.2f}')

    fig.suptitle(f'{name} , Average Pairwise Distance: {avg_pairwise_distance:.2f}, LABEL : {target_label}')
    plt.show()

