import numpy as np
import pandas as pd
from plot_logic import *


def count_label_occurrences(y_train, label):
    label_counter = 0
    for labels_list in y_train:
        if label in labels_list:
            label_counter += 1
    return label_counter


def calculate_coral_coverage(num_coral_labels, total_labels):
    return (num_coral_labels / total_labels) * 100


def calculate_means(metric_report, classifier_name, nb_image, final_metric_report):
    print("Calculate means ", classifier_name, metric_report[classifier_name])
    precision_values = [report['precision'] for report in metric_report[classifier_name]]
    recall_values = [report['recall'] for report in metric_report[classifier_name]]
    f1_values = [report['f1-score'] for report in metric_report[classifier_name]]
    support_values = [report['support'] for report in metric_report[classifier_name]]

    precision_mean = np.mean(precision_values)
    recall_mean = np.mean(recall_values)
    f1_mean = np.mean(f1_values)

    precision_std = np.std(precision_values)
    recall_std = np.std(recall_values)
    f1_std = np.std(f1_values)

    final_metric_report[classifier_name][nb_image] = {
        'precision': precision_mean,
        'recall': recall_mean,
        'f1-score': f1_mean,
        'std_precision': precision_std,
        'std_recall': recall_std,
        'std_f1-score': f1_std
    }
    return final_metric_report


def calculate_intermediate_means(metric_report, classifier_name, nb_image, final_metric_report):
    print("Intermediate report", metric_report)
    results = {}

    # Obtenir les labels
    labels = set(label for report in metric_report[classifier_name] for label in report.keys())

    # Parcourir les labels dans le rapport de métriques
    for label in labels:
        precision_values = []
        recall_values = []
        f1_values = []
        test_support_values = []
        train_support_values = []
        test_pred_values = []

        for report in metric_report[classifier_name]:
            if label in report:
                precision_values.append(report[label]['precision'])
                recall_values.append(report[label]['recall'])
                f1_values.append(report[label]['f1_score'])
                test_support_values.append(report[label]['test_support'])
                train_support_values.append(report[label]['train_support'])
                test_pred_values.append(report[label]['test_pred'])

        precision_mean = np.mean(precision_values)
        recall_mean = np.mean(recall_values)
        f1_mean = np.mean(f1_values)
        test_support_mean = np.mean(test_support_values)
        train_support_mean = np.mean(train_support_values)
        test_pred_mean = np.mean(test_pred_values)

        precision_std = np.std(precision_values)
        recall_std = np.std(recall_values)
        f1_std = np.std(f1_values)

        results[label] = {
            'precision': precision_mean,
            'recall': recall_mean,
            'f1-score': f1_mean,
            'std_precision': precision_std,
            'std_recall': recall_std,
            'std_f1-score': f1_std,
            'train_support': train_support_mean,
            'test_support': test_support_mean,
            'test_pred': test_pred_mean
        }

    # Ajouter les résultats au rapport final
    final_metric_report[classifier_name][nb_image] = results

    return final_metric_report


def comparison_cover(flat_intermediate_reports, hi_intermediate_reports, y_test, cover_labels):
    test_support = len(y_test)

    data = []

    # Parcourir les rapports plats
    for classifier_name, nb_images in flat_intermediate_reports.items():
        for nb_image, flat_report in nb_images.items():
            for label in cover_labels:
                if label in flat_report:
                    ground_truth_cover = calculate_coral_coverage(flat_report[label]['test_support'], test_support)
                    flat_cover = calculate_coral_coverage(flat_report[label]['test_pred'], test_support)

                    data.append({
                        'Classifier': classifier_name,
                        'Image Count': nb_image,
                        'Label': label,
                        'Ground Truth Coverage (%)': ground_truth_cover,
                        'Flat Coverage (%)': flat_cover,
                        'Hi Coverage (%)': None  # Placeholder
                    })

    # Parcourir les rapports hiérarchiques
    for classifier_name, nb_images in hi_intermediate_reports.items():
        for nb_image, hi_report in nb_images.items():
            for label in cover_labels:
                if label in hi_report:
                    hi_cover = calculate_coral_coverage(hi_report[label]['test_pred'], test_support)

                    # Mettre à jour les données existantes si le label et nb_image correspondent
                    for entry in data:
                        if entry['Label'] == label and entry['Image Count'] == nb_image:
                            entry['Hi Coverage (%)'] = hi_cover
                            break
                    else:
                        # Ajouter une nouvelle entrée si aucune correspondance trouvée
                        data.append({
                            'Classifier': classifier_name,
                            'Image Count': nb_image,
                            'Label': label,
                            'Ground Truth Coverage (%)': None,  # Placeholder
                            'Flat Coverage (%)': None,  # Placeholder
                            'Hi Coverage (%)': hi_cover
                        })

    df = pd.DataFrame(data)
    plot_coverage_comparison(df)
