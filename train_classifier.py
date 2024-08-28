import os
import shutil


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from mlp_head import MLPHead
from data_classes import Batch, ImageLabels
import time
import random
from img_transformation import read_csv_file
import numpy as np
import params as p
from hierarchical_utils import *

labelset_df = p.labelset_df
label_to_id = p.label_to_id
label_to_line_number = p.label_to_line_number
line_number_to_label = p.line_number_to_label

short_codes_unique = p.short_codes_unique
index_list = p.index_list


def split_dataset(batch: Batch):
    '''
    Splits dataset from patches into training
    :param batch: batch of images with patches updated with features
    :return: x_train, y_train, x_test, y_test
    '''
    features = []
    labels = []
    for image in batch.image_labels_list:
        for patch in image.patches:
            if patch.feature is None or len(patch.feature) == 0:
                raise Exception("No features encountered in patch")
            features.append(patch.feature)
            labels.append(patch.label_id)

    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        labels, test_size=0.2,
                                                        random_state=41, stratify=labels)

    return x_train, x_test, y_train, y_test


def create_classifier(train_labels: int, classifier_name: str):
    print('Train labels: ', train_labels)
    if train_labels >= 50000:
        hls, lr = (200, 100), 1e-4
    else:
        hls, lr = (100,), 1e-3
    if classifier_name == 'MLPHead-147':
        classifier = MLPHead(hidden_layers=hls, lr=lr, epochs=200, output_dim=147)
    elif classifier_name == 'MLPClassifier':
        classifier = MLPClassifier(hidden_layer_sizes=hls)
    elif classifier_name == 'MLPHead-40':
        classifier = MLPHead(hidden_layers=hls, lr=lr, epochs=200, output_dim=40)

    else:
        raise Exception("Please specify a valid classifier name")
    return classifier


def create_random_batch(input_folder: str, annotations_file: str, test_or_train: str, BATCH_SIZE=0,):
    print("Creating random batch ... ", test_or_train)
    output_folder = input_folder + '/Test'
    move = None
    if test_or_train == 'TEST':
        if not os.path.exists(output_folder):
            print("Creating random batch from scratch")
            os.makedirs(output_folder)
            move = True
            files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]
            if len(files) >= BATCH_SIZE:
                files = random.sample(files, BATCH_SIZE)
            else:
                files = random.sample(files, len(files))
        else:
            print("Loading existing batch in folder Test")
            files = [file for file in os.listdir(output_folder) if file.endswith(('.JPG', '.jpg'))]
            print("len files", len(files))
            if len(files) < BATCH_SIZE:
                input_files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]
                input_files = random.sample(input_files, BATCH_SIZE-len(files))
                move = True
    elif test_or_train == 'TRAIN':
        files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]
        if len(files) >= BATCH_SIZE > 0:
            files = random.sample(files, BATCH_SIZE)
        else:
            files = random.sample(files, len(files))

    batch = Batch()

    for file_name in files:
        if move:
            file_path = os.path.join(input_folder, file_name)
            shutil.move(file_path, output_folder)
        image_label = ImageLabels(file_name)
        batch.add_image(image_label)

    batch = read_csv_file(annotations_file, batch)

    return batch


def train_classifier_head(x_train, y_train, x_test, y_test, classifier, clf_name):
    '''
    Train classifier MLP, print accuracy score, confusion matrix and training loss calling for visualize_results
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param classifier: classifier MLP model
    :param clf_name:
    :return:
    '''
    init_time = time.time()
    enc_y_train, enc_y_test, dict_index_label = encode_labels(y_train, y_test, clf_name)
    print("Training classifier MLP...", clf_name)
    classifier.fit(x_train, enc_y_train, x_test, enc_y_test)

    y_pred, score, probas = classifier.predict(x_test)
    dec_y_pred = decode_labels(y_pred, dict_index_label)
    train_time = time.time() - init_time
    return y_test, dec_y_pred, score, classifier.loss_curve_, train_time, probas, dict_index_label


def encode_labels(y_train, y_test, clf_name):
    dict_index_label = None
    if clf_name == 'MLPHead-147':
        dict_index_label = label_to_line_number
        y_train = [label_to_line_number[label] for label in y_train]
        y_test = [label_to_line_number[label] for label in y_test]
    elif clf_name == 'MLPHead-40':
        all_labels = set(y_train + y_test)
        if len(all_labels) >= 40:
            raise ValueError("You cannot use MLPHead-30 for your dataset there is more than 40 different labels in it")

        label_to_index = {label: index for index, label in enumerate(all_labels)}
        dict_index_label = label_to_index
        y_train = [label_to_index[label] for label in y_train]
        y_test = [label_to_index[label] for label in y_test]

    return y_train, y_test, dict_index_label


def decode_labels(encoded_labels, label_to_index):
    if label_to_index is None:
        decoded_labels = encoded_labels
    else:
        index_to_label = {index: label for label, index in label_to_index.items()}
        decoded_labels = [index_to_label.get(index, "ERR") for index in encoded_labels]
    return decoded_labels


def train_classifier_sklearn(x_train, y_train, x_test, y_test, classifier):
    '''
    Train classifier MLP, print accuracy score, confusion matrix and training loss calling for visualize_results
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :param classifier: classifier MLP model
    :return:
    '''
    init_time = time.time()

    x_train_numpy = [tensor.cpu().detach().numpy() for tensor in x_train]
    y_train_numpy = np.array(y_train)
    x_test_numpy = [tensor.cpu().detach().numpy() for tensor in x_test]
    y_test_numpy = np.array(y_test)
    print("Training classifier MLP...")
    #print(index_list)
    classifier.partial_fit(x_train_numpy, y_train_numpy, classes=index_list)
    #classifier.fit(x_train_numpy, y_train_numpy)
    y_pred = classifier.predict(x_test_numpy)

    train_time = time.time() - init_time
    return y_test_numpy, y_pred, classifier.loss_curve_, train_time


def train_and_evaluate(classifier, x_train, y_train, x_test, y_test, dic=None, int_labels=None):
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    if dic is not None:
        y_pred = decode_labels(y_pred, dic)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    flat_hi_report = hierarchical_metrics_from_flat(y_pred, y_test)

    mispredicted_vectors = {}
    for i, (pred, true) in enumerate(zip(y_pred, y_test)):
        if pred != true:

            mispredicted_vectors[tuple(x_test[i])] = {
                'predicted': pred,
                'true_label': true
            }

    if int_labels is not None:
        y_hier_train = convert_label_to_hierarchy(y_train)
        y_pred_hierarchized, y_true_hierarchized = convert_labels_to_hierarchy(y_pred, y_test)
        intermediate_metrics = metrics_intermediate_labels(y_true_hierarchized, y_pred_hierarchized, y_hier_train, int_labels)
    else:
        intermediate_metrics = None
    return report, flat_hi_report, intermediate_metrics, mispredicted_vectors