import torch
import os
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle


def get_all_features(folder_path):
    all_features = []
    all_labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)

            data = torch.load(file_path)

            features = data['features'].tolist()
            labels = data['labels']

            features_np = np.array([np.array(tensor) for tensor in features])

            all_features.append(features_np)
            all_labels.extend(labels)

    all_features_np = np.concatenate(all_features, axis=0)
    return all_features_np, all_labels


def get_features_from_label_type(folder_path, label_type):
    feature_label = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)

            data = torch.load(file_path)
            features = data['features'].tolist()
            labels = data['labels']

            features_np = np.array([np.array(tensor) for tensor in features])

            indices = [i for i, x in enumerate(labels) if x == label_type]

            if len(indices) > 0:
                for indice in indices:
                    feature_label.append(features_np[indice])
    return feature_label


def get_image_labels_from_feature(feature, folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)

            data = torch.load(file_path)

            features = data['features'].tolist()
            labels = data['labels']

            # Convert features to numpy array for comparison
            features_np = np.array([np.array(tensor) for tensor in features])
            equal = []
            #Check if feature is in this image

            for f in features_np:
                equal.append(np.all(f == feature))

            index = np.where(equal)[0]
            if len(index) > 0:
                index = index[0]  # Take the first index if there are multiple matches
                return filename[:-3], index, labels

    # Return None if no matching image is found
    return None, None, None


def get_all_features_img_name(folder_path):
    all_features = []
    all_labels = []
    image_names = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)

            data = torch.load(file_path)

            features = data['features'].tolist()
            labels = data['labels']

            features_np = np.array([np.array(tensor) for tensor in features])

            all_features.append(features_np)
            all_labels.extend(labels)
            for i, label in enumerate(labels):
                image_names.append(filename + ' ' + str(i))

    all_features_np = np.concatenate(all_features, axis=0)
    return image_names, all_features_np, all_labels


def get_features(folder_path, nb_image, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    all_files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
    random.shuffle(all_files)

    selected_features = []
    selected_labels = []

    for filename in all_files[:nb_image]:
        file_path = os.path.join(folder_path, filename)
        data = torch.load(file_path)

        features = data['features'].tolist()
        labels = data['labels']

        features_np = np.array([np.array(tensor) for tensor in features])

        selected_features.extend(features_np)
        selected_labels.extend(labels)

    selected_features_np = np.array(selected_features)
    return selected_features_np, selected_labels


def load_data(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))
    features, labels = get_features(folder_path, nb_image)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1, random_state=None)
    return x_train, x_test, y_train, y_test


def load_representative_data(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))
    features, labels = get_features(folder_path, nb_image)
    label_counter = Counter(labels)
    len_test_set = len(labels)*0.1

    label_indices = {label: np.where(np.array(labels) == label)[0] for label in label_counter.keys()}

    test_indices = []
    nb_singleton_max = 5
    nb_singleton = 0
    for label, indices in label_indices.items():
        if len(indices) > 1:
            num_to_select = int(len(indices)*0.1)
            test_indices.extend(random.sample(indices.tolist(), num_to_select))
        elif len(indices) == 1:
            if nb_singleton < nb_singleton_max:
                test_indices.extend(indices.tolist())
                nb_singleton += 1

    print("TEST SIZE (patches) ", len(test_indices), " ideal test size : ", len_test_set)

    test_indices = list(set(test_indices))
    train_indices = list(set(range(len(labels))) - set(test_indices))

    features_train = [features[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]
    features_test = [features[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]

    return features_train, features_test, labels_train, labels_test


def load_representative_data_mlc(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))

    features, labels = get_features(folder_path, nb_image)

    # Filtrer les caractéristiques et labels pour éliminer ceux avec le label 'Off'
    filtered_indices = [i for i, label in enumerate(labels) if label != 'Off']
    filtered_features = [features[i] for i in filtered_indices]
    filtered_labels = [labels[i] for i in filtered_indices]

    label_counter = Counter(filtered_labels)
    len_test_set = len(filtered_labels) * 0.1

    label_indices = {label: np.where(np.array(filtered_labels) == label)[0] for label in label_counter.keys()}

    test_indices = []
    nb_singleton_max = 5
    nb_singleton = 0
    for label, indices in label_indices.items():
        if len(indices) > 1:
            num_to_select = int(len(indices) * 0.1)
            test_indices.extend(random.sample(indices.tolist(), num_to_select))
        elif len(indices) == 1:
            if nb_singleton < nb_singleton_max:
                test_indices.extend(indices.tolist())
                nb_singleton += 1

    print("TEST SIZE (patches) ", len(test_indices), " ideal test size : ", len_test_set)

    test_indices = list(set(test_indices))
    train_indices = list(set(range(len(filtered_labels))) - set(test_indices))

    features_train = [filtered_features[i] for i in train_indices]
    labels_train = [filtered_labels[i] for i in train_indices]
    features_test = [filtered_features[i] for i in test_indices]
    labels_test = [filtered_labels[i] for i in test_indices]

    return features_train, features_test, labels_train, labels_test


def load_representative_uniform_data(folder_path, nb_image=None):
    if nb_image is None:
        nb_image = len(os.listdir(folder_path))
    features, labels = get_features(folder_path, nb_image)
    label_counter = Counter(labels)
    len_test_set = int(len(labels) * 0.1)
    label_indices = {label: np.where(np.array(labels) == label)[0].tolist() for label in label_counter.keys()}

    singletons = []
    non_singletons = []

    # Sépare les singletons et les non-singletons
    for label, indices in label_indices.items():
        if len(indices) == 1:
            singletons.append(label)
        else:
            non_singletons.append(label)

    nb_singleton_max = min(10, int(len(singletons)/2))

    # Limite le nombre de singletons dans le jeu de test
    if len(singletons) > nb_singleton_max:
        singletons = random.sample(singletons, nb_singleton_max)

    test_indices = []
    for label in singletons:
        test_indices.extend(label_indices[label])

    remaining_test_set_size = len_test_set - len(test_indices)
    num_non_singletons = len(non_singletons)

    if num_non_singletons > 0:
        num_to_select_per_class = remaining_test_set_size // num_non_singletons

        for label in non_singletons:
            indices = label_indices[label]
            if len(indices) > num_to_select_per_class + 2:
                test_indices.extend(random.sample(indices, num_to_select_per_class))
            elif len(indices) < num_to_select_per_class + 2:
                test_indices.extend(random.sample(indices, int(len(indices)/2)))

    while len(test_indices) < len_test_set:
        for label in non_singletons:
            indices = label_indices[label]
            remaining_indices = list(set(indices) - set(test_indices))
            if remaining_indices:
                test_indices.append(random.choice(remaining_indices))
            if len(test_indices) >= len_test_set:
                break


    test_indices = list(set(test_indices))
    train_indices = list(set(range(len(labels))) - set(test_indices))

    features_train = [features[i] for i in train_indices]
    labels_train = [labels[i] for i in train_indices]
    features_test = [features[i] for i in test_indices]
    labels_test = [labels[i] for i in test_indices]

    return features_train, features_test, labels_train, labels_test


def load_representative_training_set(features_train, labels_train, labels_test, nb_image):
    all_labels = labels_train + labels_test
    label_counter = Counter(all_labels)
    len_train_set = nb_image * 25

    if len_train_set >= len(labels_train):
        return features_train, labels_train

    total_len_dataset = len(all_labels)

    # Calculate global proportions
    label_proportions = {label: count / total_len_dataset for label, count in label_counter.items()}

    # Find indices of each label in the training set
    label_indices = {label: np.where(np.array(labels_train) == label)[0] for label in label_counter.keys() if label in labels_train}

    train_indices = []

    nb_singleton_max = 20
    nb_singleton = 0
    for label, indices in label_indices.items():
        if len(indices) > 1 and len(train_indices) < len_train_set:
            num_to_select = int(len_train_set * label_proportions[label])
            train_indices.extend(random.sample(indices.tolist(), num_to_select))
        elif len(indices) == 1 and len(train_indices) < len_train_set:
            if nb_singleton < nb_singleton_max:
                train_indices.extend(indices)
                nb_singleton += 1

    features_train = [features_train[i] for i in train_indices]
    labels_train = [labels_train[i] for i in train_indices]

    return features_train, labels_train


def load_uniform_training_data(features_train, labels_train, nb_image):
    # Convert labels_train to a numpy array for easy indexing
    labels_train = np.array(labels_train)

    # Count the number of examples per class
    label_counter = Counter(labels_train)

    # Determine the total number of classes
    unique_labels = list(label_counter.keys())
    num_classes = len(unique_labels)

    # Calculate the number of images to select per class
    patches_per_class = nb_image*25 // num_classes

    # Initialize lists to store selected indices and final data
    selected_indices = []

    for label in unique_labels:
        # Get indices for the current class
        indices = np.where(labels_train == label)[0]

        if len(indices) >= patches_per_class:
            # Randomly sample images_per_class indices from available indices
            selected_indices.extend(random.sample(indices.tolist(), patches_per_class))
        else:
            # If not enough images for this class, take all available images
            selected_indices.extend(indices.tolist())

    if len(selected_indices) < nb_image*25:
        # If fewer images are selected than required, add more from the available indices
        additional_needed = nb_image*25 - len(selected_indices)

        # Flatten indices and select additional indices from the remaining available images
        all_indices = np.concatenate([np.where(labels_train == label)[0] for label in unique_labels])
        remaining_indices = list(set(all_indices) - set(selected_indices))

        if remaining_indices:
            selected_indices.extend(random.sample(remaining_indices, min(additional_needed, len(remaining_indices))))

    # If selected indices exceed the requested number of images, trim them
    if len(selected_indices) > nb_image*25:
        selected_indices = random.sample(selected_indices, nb_image*25)

    # Use selected indices to construct the new training set
    features_train = [features_train[i] for i in selected_indices]
    labels_train = [labels_train[i] for i in selected_indices]

    return features_train, labels_train


def save_pickle(obj, filename):
    with open(filename, 'wb') as file :
        pickle.dump(obj, file)


def load_pickle(filename):
    """Charge un objet Python depuis un fichier pickle."""
    with open(filename, 'rb') as file:
        return pickle.load(file)


def load_all_pickles(directory):
    """Charge tous les fichiers pickle (.pkl) depuis un répertoire."""
    data = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pkl'):
            # Enlever l'extension '.pkl' pour utiliser le nom de base comme clé
            key = os.path.splitext(filename)[0]
            file_path = os.path.join(directory, filename)
            if os.path.exists(file_path):
                try:
                    data[key] = load_pickle(file_path)
                except Exception as e:
                    print(f"Erreur en chargeant {file_path}: {e}")
            else:
                print(f"Le fichier {file_path} n'existe pas.")
    return data