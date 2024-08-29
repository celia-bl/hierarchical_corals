import pickle
import random
from params import code_label_full_name
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch

import params
import csv
from train_classifier import decode_labels
from collections import Counter, defaultdict
from data_classes import Batch, Point
import math
import os
from PIL import Image
from img_transformation import crop_simple
from scipy import stats


def visualize_results(y_test, y_pred, loss_curve, train_time):
    '''
    :param y_test: test labels
    :param y_pred: predicted labels
    :param loss_curve: loss_curve of the classifier
    :param train_time:
    :return:
    '''

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Train time:", train_time)
    #decoded_labels = [line_number_to_label[line_number] for line_number in y_test]
    #decoded_predictions = [line_number_to_label[line_number_pred] for line_number_pred in y_pred]
    #print("Y_Test ", y_test)
    #print("y_pred ", y_pred)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    #print("Confusion Matrix:")
    #print(conf_matrix)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    #print("Classification Report:", report)
    #plot_report(report)
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    #plt.xlabel("Prédiction")
    #plt.ylabel("Vraie étiquette")
    #plt.title("Matrice de confusion")
    #plt.show()

    # Training loss
    #plt.figure(figsize=(10, 5))
    #plt.plot(loss_curve, label='Training Loss')
    #plt.xlabel('Epochs')
    #plt.ylabel('Loss')
    #plt.title('Training Loss')
    #plt.legend()
    #plt.show()

    return accuracy, report


def results_inter_batch(results_per_batch):
    total_time = sum([batch_results[3] for batch_results in results_per_batch])
    print('Total time:', total_time)
    accuracies = []
    total_time = 0
    all_y_test = []
    all_y_pred = []

    for batch_results in results_per_batch:
        y_test, y_pred, _, train_time = batch_results
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        total_time += train_time
        y_test, y_pred, _, _ = batch_results
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    plt.plot(range(len(accuracies)), accuracies, marker='o')
    plt.title("Accuracy evolution batch")
    plt.xlabel("Batch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()

    conf_matrix = confusion_matrix(all_y_test, all_y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Prédiction")
    plt.ylabel("Vraie étiquette")
    plt.title("Matrice de confusion")
    plt.show()


def compute_probabilities_patch(batch):
    label_count_per_image = batch.labels_count_per_image()
    _, images_label_count = batch.count_labels_in_batch()
    dict_label = defaultdict(list)

    for im, labels in label_count_per_image.items():
        for label, nbr in labels.items():
            dict_label[label].append(im)

    dict_label_img = defaultdict(list)

    for label, im_list in dict_label.items():
        for img in im_list:
            dict_label_img[label].append(label_count_per_image[img])

    dict_label_avg = {}
    for label, labels_list in dict_label_img.items():
        label_dict = defaultdict(list)
        for dict_labels in labels_list:
            for label_img, value in dict_labels.items():
                label_dict[label_img].append(value)
        dict_label_avg[label] = {label_img: sum(values) / images_label_count[label] for label_img, values in label_dict.items()}

    num_groups = len(dict_label_avg)
    num_cols = 3
    num_rows = 3

    threshold = 9
    for i, (group_name, group_data) in enumerate(dict_label_avg.items()):
        remaining_groups = num_groups - i

        if i % threshold == 0:
            num_cols_current = min(num_cols, remaining_groups)

            fig, axs = plt.subplots(num_rows, num_cols, figsize=(30, 15))

        row_idx = (i % threshold) // num_cols_current
        col_idx = (i % threshold) % num_cols_current

        labels = list(group_data.keys())
        averages = list(group_data.values())
        ax = axs[row_idx, col_idx]

        ax.set_title(f'Average occurrences of labels in images {images_label_count[group_name]} with label {group_name}')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Average number of patches')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.bar(labels, averages, color='skyblue')

        plt.tight_layout()

    plt.show()


def compute_probabilities_labels(complete_batch):
    total_label_count, images_with_label_count = complete_batch.count_labels_in_batch()
    label_count_per_image = complete_batch.labels_count_per_image()
    num_images = complete_batch.batch_size()
    num_labels = len(total_label_count)
    probabilities = torch.zeros((num_labels, num_labels))
    for i, label1 in enumerate(total_label_count):
        for j, label2 in enumerate(total_label_count):
            if label1 == label2:
                probabilities[i, j] = images_with_label_count[label1]/num_images
            else:
                num_images_1 = images_with_label_count[label1]
                if num_images_1 > 0:
                    num_both_labels = sum(
                        1 for counts in label_count_per_image.values() if label1 in counts and label2 in counts)
                    probabilities[i, j] = num_both_labels/num_images_1

    plot_probabilities(probabilities, list(total_label_count.keys()), images_with_label_count)
    proba_tuple = probabilities_tuple_labels(complete_batch)
    plot_dict_graph(proba_tuple, images_with_label_count)


def probabilities_tuple_labels(complete_batch):
    total_label_count, images_with_label_count = complete_batch.count_labels_in_batch()
    label_count_per_image = complete_batch.labels_count_per_image()
    # dict : {label 1 : {(label2,3):prob}, label 2 : {(label3,4):prob} }
    high_prob_combinations = {}
    usual_labels = ['TFL', 'Turf_sand', 'Sand']
    for i, label1 in enumerate(total_label_count):
        for j, label2 in enumerate(total_label_count):
            for k, label3 in enumerate(total_label_count):
                if (label2, label3).count(label1) <= 1 and label2 != label3 and (label3 != label1 and label2 != label1) and label2 not in usual_labels and label3 not in usual_labels:
                    num_images_1 = images_with_label_count[label1]
                    if num_images_1 > 0:
                        num_tuple_labels = sum(
                            1 for counts in label_count_per_image.values() if label1 in counts and label2 in counts and label3 in counts
                        )
                        if num_tuple_labels/num_images_1 > 0.5:
                            if label1 not in high_prob_combinations:
                                high_prob_combinations[label1] = {}
                            high_prob_combinations[label1].update({(label2, label3): num_tuple_labels/num_images_1})
    #print("HPC : ", high_prob_combinations)
    return high_prob_combinations


def plot_dict_graph(dictionary, images_with_label_count):
    labels = list(dictionary.keys())
    num_labels = len(labels)

    unique_tuples = set()
    for label, tuples_prob in dictionary.items():
        for tuple_labels in tuples_prob.keys():
            unique_tuples.add(tuple(sorted(tuple_labels)))

    #print(unique_tuples)
    probabilities = np.zeros((num_labels, len(unique_tuples)))

    for i, label1 in enumerate(dictionary.keys()):
        for j, tuple_label in enumerate(unique_tuples):
            if tuple_label in dictionary[label1]:
                probabilities[i, j] = dictionary[label1][tuple_label]
            elif tuple_label[::-1] in dictionary[label1]:
                probabilities[i, j] = dictionary[label1][tuple_label[::-1]]
            else:
                probabilities[i, j] = 0
    num_rows, num_cols = probabilities.shape
    #print(probabilities)
    plt.figure(figsize=(0.7*len(unique_tuples), 0.8*num_rows))
    plt.imshow(probabilities, cmap='viridis', interpolation='nearest')

    unique_tuples_as_list = [str(t) for t in unique_tuples]
    plt.yticks(np.arange(num_labels), labels)
    plt.xticks(range(len(unique_tuples)), unique_tuples_as_list, rotation=90)

    for i, label in enumerate(labels):
        if images_with_label_count[label] < 10:
            plt.gca().get_yticklabels()[i].set_color('cornflowerblue')
        else:
            plt.gca().get_yticklabels()[i].set_color('black')

    for j, tuple_str in enumerate(unique_tuples_as_list):
        label_tuple = eval(tuple_str)
        label1 = label_tuple[0]
        label2 = label_tuple[1]
        if images_with_label_count[label1] < 10 or images_with_label_count[label2] < 10 :
            plt.gca().get_xticklabels()[j].set_color('cornflowerblue')
    for i in range(len(unique_tuples)):
        plt.axvline(x=i - 0.5, color='black', linewidth=0.5)

    for i in range(len(labels)):
        plt.axhline(y=i - 0.5, color='black', linewidth=0.5)

    for i in range(len(labels)):
        for j in range(len(unique_tuples)):
            if probabilities[i][j] >= 0.5:
                plt.text(j, i, round(probabilities[i, j].item(), 2), ha='center', va='center', color='red')

    plt.xlabel('Tuple labels')
    plt.ylabel('Label y')
    plt.title('Probabilities between Label y and tuple labels')
    plt.colorbar(label='Probability', shrink=0.5)
    plt.tight_layout()
    plt.show()


def plot_probabilities(probabilities, labels, images_with_label_count):
    sorted_indices = np.argsort([images_with_label_count[label] for label in labels])
    labels = [labels[i] for i in sorted_indices]
    probabilities = probabilities[sorted_indices][:, sorted_indices]
    if len(labels) < 40 :
        plt.figure(figsize=(len(labels)*0.45, len(labels)*0.35))
    else:
        plt.figure(figsize=(len(labels)*0.2, len(labels)*0.15))
    plt.imshow(probabilities, cmap='viridis', interpolation='nearest')

    plt.yticks(np.arange(len(labels)), labels)
    plt.xticks(np.arange(len(labels)), labels, rotation=90)

    for i, label in enumerate(labels):
        if images_with_label_count[label] < 10:
            plt.gca().get_yticklabels()[i].set_color('cornflowerblue')
            plt.gca().get_xticklabels()[i].set_color('cornflowerblue')
        else:
            plt.gca().get_yticklabels()[i].set_color('black')
            plt.gca().get_xticklabels()[i].set_color('black')

    for i in range(len(labels)):
        for j in range(len(labels)):
            if probabilities[i][j] >= 0.4:
                if len(labels) < 40:
                    plt.text(j, i, round(probabilities[i, j].item(), 2), ha='center', va='center', color='red')
                else:
                    plt.text(j, i, round(probabilities[i, j].item(), 2), ha='center', va='center', color='red', fontsize= 5)
        plt.axvline(x=i - 0.5, color='black', linewidth=0.5)
        plt.axhline(y=i - 0.5, color='black', linewidth=0.5)
        plt.gca().add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=None, edgecolor='red', linewidth=1))

    plt.colorbar()
    plt.xlabel('Labels')
    plt.ylabel('Labels')
    plt.title('Probabilities of Label y given Label x nb_im_label_x_y/nb_im_label_x')
    legend_elements = [
        plt.Line2D([0], [0], color='cornflowerblue', lw=0.5, label='Labels < 20 images'),
        plt.Line2D([0], [0], color='red', lw=0.5, label='probability of label occurrence in dataset nb_im_label/nb_im')
    ]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.15, 1.15))
    plt.show()


def visualize_final(accuracies, times, BATCH_SIZE_LIST):
    plt.figure()
    for network, network_accuracies in accuracies.items():
        for classifier, accuracy in network_accuracies.items():
            plt.scatter(BATCH_SIZE_LIST, accuracy, label=f"{network} - {classifier}")
    plt.xlabel('Number of training images')
    plt.ylabel('Accuracy')
    plt.title('Accuracy based on the number of images in training set \n for Different Neural Networks/Classifiers')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    for network, network_times in times.items():
        for classifier, time in network_times.items():
            plt.scatter(BATCH_SIZE_LIST, time, label=f"{network} - {classifier}")
    plt.xlabel('Number of training images')
    plt.ylabel('Times')
    plt.title('Times based on the number of images in training set \n for Different Neural Networks/Classifiers')
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_average_scores_per_label(y_pred, scores):
    df = pd.DataFrame({'label': y_pred, 'score': scores})
    average_scores_labels = df.groupby('label')['score'].mean().to_dict()
    return average_scores_labels


def sorting_labels(train, test):
    if isinstance(train, Batch) and isinstance(test, Batch):
        total_label_count_tr, _ = train.count_labels_in_batch()
        total_label_count_te, _ = test.count_labels_in_batch()

        sorted_labels_tr = sorted(total_label_count_tr.keys(), key=lambda x: total_label_count_tr.get(x, 0))
        sorted_labels_te = sorted(total_label_count_te.keys(), key=lambda x: total_label_count_te.get(x, 0))

        sorted_labels = sorted(set(sorted_labels_tr + sorted_labels_te), key=lambda x: total_label_count_tr.get(x, 0))
        sorted_labels.append('Total')
        total_counts = [total_label_count_tr.get(label, 0) for label in sorted_labels]
        len_test_labels = len(set(sorted_labels_te))

    elif isinstance(train, list) and isinstance(test, list):
        labels = train + test
        label_counts = Counter(labels)
        sorted_labels = [label for label, _ in sorted(label_counts.items(), key=lambda x: x[1])]
        sorted_labels.append('Total')
        total_counts = [label_counts.get(label, 0) for label in sorted_labels]
        len_test_labels = len(set(test))
    else:
        raise ValueError('X, y nor batch type nor list type')

    return sorted_labels, total_counts, len_test_labels


def complete_analyze(train_elements, test_elements, report, y_pred, score):
    print("Complete analyse with training batch and testing batch")
    # total_label_count = dict with label : occurence of this label in the batch
    sorted_labels, total_counts, len_test_labels = sorting_labels(train_elements, test_elements)

    # average_scores_label = dict with label : mean of the scores of this label
    average_scores_labels = calculate_average_scores_per_label(y_pred, score)
    average_score_per_label = [average_scores_labels.get(label, 0) for label in sorted_labels[:-1]]
    average_score_per_label.append(0)

    # analyze of report
    accuracy_per_label = []
    f1_per_label = []
    # support = number of occurence of this label in the test batch
    support_per_label = []

    for label in sorted_labels[:-1]:
        report_label = report.get(str(label), {})
        accuracy_per_label.append(report_label.get('precision', 0))
        f1_per_label.append(report_label.get('f1-score', 0))
        support_per_label.append(report_label.get('support', 0))

    accuracy_per_label.append(report['accuracy'])
    f1_per_label.append(report['macro avg']['f1-score'])
    support_per_label.append(report['macro avg']['support'])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 13), sharex=True)
    bar_width = 0.35

    index = range(len(sorted_labels))

    ax1.bar(index, total_counts, bar_width, label='Number of Patches', color='orange')
    ax1.set_ylabel('Number of patches in train dataset', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.set_xticks(index)
    ax1.set_xticklabels(sorted_labels, rotation=45, ha='right')

    ax2.bar(index, f1_per_label, bar_width, label='F1 per label', color='blue')
    ax2.set_ylabel('F1 per label', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.spines['bottom'].set_position(('axes', 0))
    ax2.spines['top'].set_color('black')
    ax2.xaxis.set_ticks_position('top')
    ax2.invert_yaxis()

    for i, v in enumerate(total_counts):
        ax1.text(i, v + max(total_counts)/40, str(v), color='orange', ha='center', va='top')

    for i, v in enumerate(support_per_label):
        ax2.text(i, f1_per_label[i] + 0.01, str(v), color='blue', ha='center', va='top')

    for i, v in enumerate(average_score_per_label):
        ax1.text(i, -max(total_counts)/25, str(round(v, 2)), color='black', ha='center', va='bottom')

    total_labels_text = "Total number of labels in train set : {}".format(len(set(total_counts)))
    total_labels = "Total number of annotations in train set : {}".format(sum(total_counts))
    custom_legend = [plt.Line2D([0], [0], color='orange', lw=4),
                     plt.Line2D([0], [0], color='black', lw=0.5),
                     plt.Line2D([0], [0], color='none'),
                     plt.Line2D([0], [0], color='none')]
    ax1.legend(custom_legend,
               ['Number of Patches', 'Average score predicted with .predict(x_test)', total_labels_text, total_labels],
               loc='upper left')

    total_labels_test = "Total number of labels in test set : {}".format(len_test_labels)
    custom_legend_test = [plt.Line2D([0], [0], color='blue', lw=4),
                          plt.Line2D([0], [0], color='blue', lw=0.5),
                          plt.Line2D([0], [0], color='none')]
    ax2.legend(custom_legend_test, ['F1 per label',
                                    'Number of patches with this label in test set',
                                    total_labels_test], loc='lower left')

    plt.suptitle('Data analysis labels in train and test set', y=0.98)

    plt.tight_layout()
    plt.show()


def analyze_labels(batch, name):
    print("ANALYZE NUMBER OF IMAGES ", batch.batch_size())
    nb_labels_per_image = []
    for image_name, label_count in batch.labels_count_per_image().items():
        #print("LABEL COUNT DE I ", label_count, "LEN ", len(label_count))
        nb_labels_per_image.append(len(label_count))

    #print(nb_labels_per_image)
    mean_labels_image, std_labels_image, first_quartile_labels_image, third_quartile_labels_image = calculate_statistics(nb_labels_per_image)
    plot_statistics(nb_labels_per_image, mean_labels_image, std_labels_image, first_quartile_labels_image,
                    third_quartile_labels_image, 'Number of Labels per Image', 'Number of Images',
                    'Label Count Per Image Statistics '+name)

    total_label_count, images_with_label_count = batch.count_labels_in_batch()

    #plot_labels(total_label_count, images_with_label_count, 'Number of patches', 'Number of images',
    #            'Frequency of labels occurence ' + name)
    #plot_labels(images_with_label_count, 'Label', 'Number of Images', 'Number of images with each label')
    plot_labels_table(total_label_count)

def plot_labels(total_label_count, images_with_label_count, y1_name, y2_name, title):
    labels = list(total_label_count.keys())
    total_counts = list(total_label_count.values())

    image_counts = [-images_with_label_count[label] for label in labels]
    average_patch_per_image = [total_label_count[label] / images_with_label_count[label] for label in labels]

    labels, total_counts, image_counts, average_patch_per_image = zip(
        *sorted(zip(labels, total_counts, image_counts, average_patch_per_image), key=lambda x: x[1]))
    bar_width = 0.35
    index = np.arange(len(labels))

    if len(labels):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(33, 13), sharex=True)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 13), sharex=True)

    ax1.bar(index, total_counts, bar_width, label='Number of Patches', color='orange')
    ax1.set_ylabel(y1_name, color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    #plt.xticks(index, labels, rotation=45, ha='right')
    ax1.set_xticks(index)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    ax2.bar(index, image_counts, bar_width, label='Number of Images', color='blue')
    ax2.set_ylabel(y2_name, color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    ax2.spines['bottom'].set_position(('axes', 0))
    ax2.spines['top'].set_color('black')
    ax2.xaxis.set_ticks_position('top')

    for i, v in enumerate(total_counts):
        ax1.text(i, v + max(total_counts)/40, str(v), color='orange', ha='center', va='top')

    for i, v in enumerate(image_counts):
        ax2.text(i, v - max(image_counts)/80, str(v), color='blue', ha='center', va='top')

    for i, v in enumerate(average_patch_per_image):
        ax1.text(i, -max(total_counts)/25, str(round(v, 2)), color='black', ha='center', va='bottom')

    total_labels_text = "Total number of labels: {}".format(len(index))
    total_labels = "Total number of annotations: {}".format(sum(total_counts))
    custom_legend = [plt.Line2D([0], [0], color='orange', lw=4),
                     plt.Line2D([0], [0], color='black', lw=0.5),
                     plt.Line2D([0], [0], color='none'),
                     plt.Line2D([0], [0], color='none')]
    ax1.legend(custom_legend, ['Number of Patches', 'Average number of patches per image', total_labels_text, total_labels], loc='upper left')

    plt.suptitle(title, y=0.98)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_labels_table(total_label_count):
    # Convertir les données en listes triées par ordre décroissant
    labels = list(total_label_count.keys())
    total_counts = list(total_label_count.values())

    # Trier par ordre décroissant
    labels, total_counts = zip(*sorted(zip(labels, total_counts), key=lambda x: x[1], reverse=True))

    # Remplacer les codes par les noms complets
    full_names = [code_label_full_name.get(label, label) for label in labels]

    # Diviser les données en deux colonnes
    half = len(full_names) // 2
    full_names_col1, full_names_col2 = full_names[:half], full_names[half:]
    counts_col1, counts_col2 = total_counts[:half], total_counts[half:]

    # Créer une figure pour afficher le tableau
    fig, ax = plt.subplots(figsize=(40, len(full_names) * 0.7))  # Ajuster la largeur en fonction du nombre de labels
    ax.axis('off')

    # Créer le tableau avec deux colonnes
    table_data = []
    max_rows = max(len(full_names_col1), len(full_names_col2))

    for i in range(max_rows):
        row = []
        if i < len(full_names_col1):
            row.append(full_names_col1[i])
            row.append(counts_col1[i])
        else:
            row.append('')
            row.append('')

        if i < len(full_names_col2):
            row.append(full_names_col2[i])
            row.append(counts_col2[i])
        else:
            row.append('')
            row.append('')

        table_data.append(row)

    col_labels = ["Labels", "Number of Patches", "Labels", "Number of Patches"]
    table = ax.table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='center')

    # Ajuster la taille de la police et la largeur des colonnes
    table.auto_set_font_size(False)
    table.set_fontsize(40)
    table.auto_set_column_width([0, 1, 2, 3])

    # Ajuster la hauteur des cellules
    for key, cell in table.get_celld().items():
        cell.set_height(0.04)  # Ajuster la hauteur des cellules (0.04 est un exemple, ajustez selon vos besoins)

    # Afficher le tableau
    plt.show()


def calculate_statistics(stat_list):
    mean = np.mean(stat_list)
    std = np.std(stat_list)
    first_quartile = np.percentile(stat_list, 25)
    third_quartile = np.percentile(stat_list, 75)
    return mean, std, first_quartile, third_quartile


def plot_statistics(stat_list, mean, std, first_quartile, third_quartile, x_name, y_name, title):
    plt.figure(figsize=(10, 6))
    plt.hist(stat_list, bins='auto', alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=mean, color='red', linestyle='dashed', linewidth=1, label='Mean')
    plt.axvline(x=mean - std, color='orange', linestyle='dashed',
                linewidth=1, label='Mean - Std Dev')
    plt.axvline(x=mean + std, color='orange', linestyle='dashed',
                linewidth=1, label='Mean + Std Dev')
    plt.axvline(x=first_quartile, color='green', linestyle='dashed', linewidth=1, label='1st Quartile')
    plt.axvline(x=third_quartile, color='purple', linestyle='dashed', linewidth=1,
                label='3rd Quartile')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_report(report_dict):
    print('Plotting report...')
    labels = []
    accuracy_per_label = []
    support_per_label = []

    for key, value in report_dict.items():
        if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg':
            labels.append(key)
            accuracy_per_label.append(value.get('precision', 0))
            support_per_label.append(value.get('support', 0))
    labels.append('Total')
    accuracy_per_label.append(report_dict['accuracy'])
    support_per_label.append(report_dict['macro avg']['support'])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))

    bar_width = 0.35
    bars = ax.bar(x, accuracy_per_label, bar_width, label='Accuracy', color='blue')

    for i, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, support_per_label[i],
                ha='center', va='bottom')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    ax.set_xlabel('Label')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy per Label with number of patches in the test dataset')

    ax.legend()

    plt.tight_layout()
    plt.show()


def analyze_probas(probabilities, dict_dec, y_test, feature_list, y_pred):
    counts = count_probas(probabilities)
    plot_counts_occurrences(counts)
    labels_info = get_labels_info(probabilities, dict_dec, y_test)
    #with open('labels_info.pkl', 'wb') as f:
    #    pickle.dump(labels_info, f)
    #with open('labels_info.pkl', 'rb') as f:
    #    labels_info = pickle.load(f)
    plot_analyze_probas(labels_info)
    #plot_hesitating_scores(labels_info)
    plot_usual_errors(labels_info, feature_list, y_test, y_pred)


def plot_hesitating_scores(labels_info):
    print('Plotting hesitating scores...')
    plt.figure(figsize=(12, 12))
    for i, true_label in enumerate(labels_info):
        if labels_info[true_label]['unique_label_hesitating']:
            label_predicted = list(labels_info[true_label]['unique_label_predicted'])
            avg_predicted = labels_info[true_label]['avg_predicted']
            label_hesitated = list(labels_info[true_label]['unique_label_hesitating'])
            avg_hesitated = labels_info[true_label]['avg_hesitating_scores']
            for j in range(len(label_predicted)):
                plt.scatter(true_label, avg_predicted[j], color='blue')
                plt.text(true_label, avg_predicted[j]+0.013, label_predicted[j], ha='center', va='center')
            for k in range(len(label_hesitated)):
                plt.scatter(true_label, avg_hesitated[k], color='red')
                plt.text(true_label, avg_hesitated[k]+0.013, label_hesitated[k], ha='center', va='center')

    plt.scatter([], [], color='blue', marker='o', label='Predicted')
    plt.scatter([], [], color='red', marker='s', label='Hesitated')
    plt.legend(loc='upper right')
    plt.xlabel('True Label')
    plt.ylabel('Average Score')
    plt.title('Average Scores by True Label')
    plt.show()


def count_probas(probabilities):
    counts = []
    for probas in probabilities:
        count = 0
        for proba in probas:
            if proba > (torch.max(probas) - 0.1):
                count += 1
        counts.append(count)
    return counts


def plot_counts_occurrences(counts):
    counts_occurrences = Counter(counts)
    plt.bar(list(counts_occurrences.keys()), list(counts_occurrences.values()))
    plt.xlabel('Number of probas > score_max - 0.1')
    plt.ylabel('Number of patches in test set')
    plt.title('Occurrences of high probas in probabilities from test set \n (high probas = probas > score_max - 0.1)')
    plt.grid(True)
    plt.show()


def get_labels_info(probabilities, dict_dec, y_test):
    print('Making dictionary from probabilities...')
    labels_info = {}
    for i, probas in enumerate(probabilities):
        true_label = y_test[i]
        if true_label not in labels_info:
            labels_info[true_label] = {
                'label_predicted': [],
                'unique_label_predicted': set(),
                'score_label_predicted': [],
                'avg_predicted': [],
                'occurence_predicted': [],
                'score_true_label': [],
                'avg_true_score': 0.0,
                'hesitating_labels': [],
                'unique_label_hesitating': set(),
                'avg_hesitating_scores': [],
                'hesitating_scores': [],
                'occurence_hesitating': []
            }

        score_max, predicted = torch.max(probas, 0)
        labels_info[true_label]['label_predicted'].append(predicted)
        labels_info[true_label]['unique_label_predicted'].add(decode_labels([int(predicted)], dict_dec)[0])
        labels_info[true_label]['score_label_predicted'].append(score_max)
        if probas[dict_dec[true_label]] != score_max:
            labels_info[true_label]['score_true_label'].append(float(probas[dict_dec[true_label]]))
        #for lab, proba in enumerate(probas):
        #    score_max = max(probas)
        #    if proba > (score_max - 0.1) and proba != score_max:
        #        labels_info[true_label]['hesitating_labels'].append(lab)
        #        labels_info[true_label]['unique_label_hesitating'].add(decode_labels([int(lab)], dict_dec)[0])
        #        labels_info[true_label]['hesitating_scores'].append(proba)

    for true_labels, infos in labels_info.items():
        if len(infos['score_true_label']) > 0:
            avg_score = sum(infos['score_true_label']) / len(infos['score_true_label'])
            infos['avg_true_score'] = avg_score
        for un_label in infos['unique_label_predicted']:
            indexes = [i for i, x in enumerate(infos['label_predicted']) if int(x) == dict_dec[un_label]]
            infos['occurence_predicted'].append(len(indexes))
            avg_score_pred = sum(infos['score_label_predicted'][i] for i in indexes) / len(indexes)
            infos['avg_predicted'].append(avg_score_pred)
        #for un_h_label in infos['unique_label_hesitating']:
        #    if un_h_label != 'ERR':
        #        indexes = [i for i, x in enumerate(infos['hesitating_labels']) if int(x) == dict_dec[un_h_label]]
        #        infos['occurence_hesitating'].append(len(indexes))
        #        avg_score_hesitating = sum(infos['hesitating_scores'][i] for i in indexes) / len(indexes)
        #        infos['avg_hesitating_scores'].append(avg_score_hesitating)
        #        del infos['hesitating_labels']
        #    else:
        #        infos['avg_hesitating_scores'].append(0)
    return labels_info


def get_patch_ft(feature):
    print('Getting Patch from Feature ... ')
    indices = []
    for file in os.listdir('.features/efficientnet-b0_features'):
        tensor = torch.load(os.path.join('.features/efficientnet-b0_features', file))
        tensor = tensor['features']
        for i in range(tensor.size(0)):
            current_vector = tensor[i].to('cuda')
            if torch.all(current_vector == feature):
                indices.append(i)
                image_name, extension = os.path.splitext(file)
                full_image_name = image_name + '.JPG'
    if len(indices) > 1:
        print(indices, image_name)
        #raise ValueError('More than one feature equal for this image and feature')
        print("feature with more than one match")
        return Image.new('RGB', (224, 224)), 0, (0, 0), 0
    csv_file = params.annotations_file
    rows_columns_labels = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Name'] == full_image_name:
                rows_columns_labels.append({
                    'row': int(row['Row']),
                    'column': int(row['Column']),
                    'label': row['Label']
                })
    print(full_image_name)
    print(len(rows_columns_labels))
    if len(rows_columns_labels) < 25:
        raise ValueError('Not enough patch in images ?')
    patch_row = rows_columns_labels[indices[0]]['row']
    patch_column = rows_columns_labels[indices[0]]['column']
    patch_label = rows_columns_labels[indices[0]]['label']
    path_img = os.path.join(params.path_images_batches, full_image_name)
    im = Image.open(path_img)
    patch = crop_simple(im, Point(patch_row, patch_column), 224)
    return patch, patch_label, (patch_row, patch_column), full_image_name


def plot_usual_errors(labels_infos, feature_list, label_list, y_pred):
    #existe plus le label_predicted
    print('Plotting images of usual errors')
    #list_label_plot = ['TCC', 'TFL', 'Turf_sand', 'PAL', 'ZOS', 'CNA/CC']
    list_label_plot = ['Unk']
    for label in list_label_plot:
        predicted_labels = []
        patches_images = []
        tuple_list = []
        full_image_name_list = []
        label_predicted_list = list(labels_infos[label]['unique_label_predicted'])
        for i, predicted_label in enumerate(label_predicted_list):
            if labels_infos[label]['occurence_predicted'][i] > 1:
                print(predicted_label)
                indices = [i for i, label_pred in enumerate(y_pred) if (label_pred == predicted_label and label_list[i] == label)]
                random_indices = random.sample(indices, min(10, len(indices)))
                print(len(random_indices))
                features = [feature_list[i] for i in random_indices]
                labels_predicted = [label_list[i] for i in random_indices]

                for j, feature in enumerate(features):
                    patch, patch_label, tuple_point, full_image_name = get_patch_ft(feature)
                    if patch_label == labels_predicted[j]:
                        patches_images.append(patch)
                        predicted_labels.append(predicted_label)
                        full_image_name_list.append(full_image_name)
                        tuple_list.append(tuple_point)
                    else:
                        #raise ValueError('Label mismatch')
                        print('label mismatch')
        print('Predicted_labels: ', predicted_labels)
        print('Length of patches: ', len(patches_images))
        fig, ax = plt.subplots(10, len(set(predicted_labels)), figsize=(len(set(predicted_labels))*2, 2 * 10))

        for j, predicted_label in enumerate(list(set(predicted_labels))):
            indices = [i for i, label in enumerate(predicted_labels) if label == predicted_label][:10]

            for i, index in enumerate(indices):
                ax[i, j].imshow(patches_images[index])
                ax[i, j].set_title(predicted_label)
                ax[i, j].axis('off')
                ax[i, j].text(0.5, 1.185, f'{full_image_name_list[index]}', horizontalalignment='center', verticalalignment='center',
                              transform=ax[i, j].transAxes, fontsize=5)
                ax[i, j].text(0.5, 1.009, f'{tuple_list[index]}', horizontalalignment='center', verticalalignment='center',
                              transform=ax[i, j].transAxes, fontsize=5)

        fig.suptitle('Samples of patches predicted label when the true label is \n ' + label, fontsize=16)
        plt.tight_layout()
        plt.show()


def plot_analyze_probas(labels_info):
    print('Plotting probas graph...')
    plt.figure(figsize=(24, 18))
    label_colors = {}
    predicted_labels = set()

    for true_label, infos in labels_info.items():
        for predicted_label in infos['unique_label_predicted']:
            predicted_labels.add(predicted_label)

    color_index = 0
    cmap = plt.cm.get_cmap('tab20', len(predicted_labels))

    for predicted_label in predicted_labels:
        label_colors[predicted_label] = cmap(color_index)
        color_index += 1

    legend_used = set()
    x_positions = range(len(labels_info))
    for i, (true_label, infos) in enumerate(labels_info.items()):
        true_label_position = x_positions[i]
        for predicted_label, avg_score, occurrences in zip(infos['unique_label_predicted'], infos['avg_predicted'], infos['occurence_predicted']):
            color = label_colors[predicted_label]
            if predicted_label not in legend_used:
                plt.scatter(true_label, avg_score, color=color, label=f'{predicted_label}', alpha=0.8, s=2*occurrences+20)
                legend_used.add(predicted_label)
            else:
                plt.scatter(true_label, avg_score, color=color, alpha=0.8, s=2*occurrences+20)


            if occurrences < 15:
                plt.text(true_label_position + 0.05, avg_score, str(occurrences), color='black', fontsize=8)
            elif 40 >= occurrences > 15:
                plt.text(true_label_position + 0.15, avg_score, str(occurrences), color='black', fontsize=8)
            else:
                plt.text(true_label_position + 0.25, avg_score, str(occurrences), color='black', fontsize=8)

            #if predicted_label == true_label:
            #    plt.scatter(true_label, avg_score, color=color, s=100)

        avg_true_score = infos['avg_true_score']
        plt.scatter(true_label, avg_true_score, color='red', marker='x')
        plt.text(true_label_position, avg_true_score, str(len(infos['score_true_label'])), color='black')

    plt.xlabel('True Label')
    plt.ylabel('Average Score of Predicted Label')
    plt.title('Average Score of Predicted Label vs True Label')
    custom_legend = [plt.scatter([], [], color=label_colors[label], label=label, marker='o', s=40) for label in legend_used]
    plt.legend(handles=custom_legend, title='Predicted Label', bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()


def plot_label_distribution(y):
    # Convertir y en un tableau numpy pour faciliter le traitement
    y_array = np.array(y)

    # Calculer la distribution des labels
    unique_labels, counts = np.unique(y_array, return_counts=True)

    # Trier les labels par nombre d'occurrences (de manière décroissante)
    sorted_indices = np.argsort(counts)[::-1]  # Tri décroissant
    sorted_labels = unique_labels[sorted_indices]
    sorted_counts = counts[sorted_indices]

    # Calculer la skewness et la kurtosis
    skewness = stats.skew(sorted_counts)
    kurtosis = stats.kurtosis(sorted_counts)

    # Créer l'histogramme
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_labels, sorted_counts, color='skyblue', edgecolor='black')

    # Ajouter des annotations pour skewness et kurtosis
    textstr = '\n'.join((
        f'Skewness: {skewness:.2f}',
        f'Kurtosis: {kurtosis:.2f}'
    ))

    # Ajouter une annotation dans le graphique
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Labels')
    plt.ylabel('Number of Occurrences')
    plt.title('Distribution of Labels in y_test')
    plt.xticks(sorted_labels)  # Afficher les labels sur l'axe des x
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_avg_distrib(average_label_counts):
    # Extraire les labels et les moyennes des occurrences
    labels = list(average_label_counts.keys())
    avg_counts = list(average_label_counts.values())

    # Convertir en arrays pour faciliter le traitement
    labels_array = np.array(labels)
    avg_counts_array = np.array(avg_counts)

    # Trier les labels par nombre moyen d'occurrences (de manière décroissante)
    sorted_indices = np.argsort(avg_counts_array)[::-1]  # Tri décroissant
    sorted_labels = labels_array[sorted_indices]
    sorted_avg_counts = avg_counts_array[sorted_indices]

    # Calculer la skewness et la kurtosis
    skewness = stats.skew(sorted_avg_counts)
    kurtosis = stats.kurtosis(sorted_avg_counts)

    # Créer l'histogramme
    plt.figure(figsize=(20, 6))
    plt.bar(sorted_labels, sorted_avg_counts, color='skyblue', edgecolor='black')

    # Ajouter des annotations pour skewness et kurtosis
    textstr = '\n'.join((
        f'Skewness: {skewness:.2f}',
        f'Kurtosis: {kurtosis:.2f}'
    ))

    # Ajouter une annotation dans le graphique
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                   fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Labels')
    plt.ylabel('Average Number of Occurrences')
    plt.title('Average Distribution of Labels Across Runs')
    plt.xticks(sorted_labels)  # Afficher les labels sur l'axe des x
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()