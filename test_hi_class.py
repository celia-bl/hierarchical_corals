from sklearn.metrics import classification_report

import hierarchy_dictionary as hi_dic
import torch
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from hiclass.metrics import f1, precision, recall
import numpy as np
import os
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from memory_profiler import profile
from line_profiler import LineProfiler
import cProfile

from collections import Counter


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


def custom_train_test_split(features, labels, test_size=0.2, random_state=None):
    """
    Divise les features et les labels en ensembles d'entraînement et de test.

    Args:
    - features : les caractéristiques (X)
    - labels : les étiquettes (y)
    - test_size : la proportion des données à utiliser pour l'ensemble de test
    - random_state : la seed pour la reproductibilité des résultats

    Returns:
    - x_train, x_test, y_train, y_test : les ensembles d'entraînement et de test pour les features et les labels
    """
    np.random.seed(random_state)

    indices = np.arange(len(features))
    np.random.shuffle(indices)

    num_test_samples = int(test_size * len(features))

    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    labels_np = np.array(labels, dtype=object)

    x_train, x_test = features[train_indices], features[test_indices]
    y_train, y_test = np.take(labels_np, train_indices), np.take(labels_np, test_indices)

    return x_train, x_test, y_train, y_test


def extract_last_element(list_of_lists):
    return [sublist[-1] for sublist in list_of_lists]


def normal_metrics(y_pred, y_true):
    new_y_pred = []
    for list_label in y_pred :
        while list_label[-1] == '':
            list_label = list_label[:-1]

        new_y_pred.append(list_label.tolist())

    y_pred_reshape = np.array(new_y_pred, dtype=object)

    y_pred_last = extract_last_element(y_pred_reshape)
    y_true_last = extract_last_element(y_true)

    # Combine y_true_last and y_pred_last to get all unique classes
    all_classes = set(y_true_last + y_pred_last)

    # Sort the classes alphabetically
    class_names = sorted(all_classes)

    # Calculate accuracy
    accuracy = accuracy_score(y_true_last, y_pred_last)
    print("Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_true_last, y_pred_last, target_names=class_names, zero_division=0, output_dict=True)

    print("Classification Report:\n", report)

    plt.figure(figsize=(15, 15))
    ConfusionMatrixDisplay.from_predictions(y_true_last, y_pred_last, ax=plt.gca(), xticks_rotation=90)

    plt.show()
    return report


def plot_comparative_graph(flat_weighted_avg, hi_weighted_avg, hiclass_mlp_scores):
    metrics = ['F1 Score', 'Recall', 'Precision']
    mlp_scores = [flat_weighted_avg['f1-score'], flat_weighted_avg['recall'], flat_weighted_avg['precision']]
    hi_mlp_scores = [hi_weighted_avg['f1-score'], hi_weighted_avg['recall'], hi_weighted_avg['precision']]
    hiclass_mlp_scores = [hiclass_mlp_scores['f1'], hiclass_mlp_scores['recall'], hiclass_mlp_scores['precision']]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots()
    bars1 = ax.bar(x - width, mlp_scores, width, label='Weighted Avg Flat MLP')
    bars2 = ax.bar(x, hi_mlp_scores, width, label='Weighted Avg HI MLP')
    bars3 = ax.bar(x + width, hiclass_mlp_scores, width, label='HiScores MLP PPNode')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Comparison of Weighted Avg and HiClass Scores')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # décalage vertical
                        textcoords="offset points",
                        ha='center', va='bottom')

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.show()


#@profile
def main():
    folder_path = '.features/efficientnet-b0_features'

    mlp_node = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    classifier_mlp = LocalClassifierPerParentNode(local_classifier=mlp_node)

    features, labels = get_all_features(folder_path)
    print('Splitting data set')
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=None)

    print('Converting labels into hierarchical labels')
    y_hier_train = np.array([hi_dic.find_hierarchy_path(label) for label in y_train], dtype=object)
    y_hier_test = np.array([hi_dic.find_hierarchy_path(label) for label in y_test], dtype=object)

    clf_mlp = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000)

    print('Fitting MLP Classifier')
    clf_mlp.fit(x_train, y_train)
    y_pred_mlp = clf_mlp.predict(x_test)
    report_mlp = classification_report(y_test, y_pred_mlp, zero_division=0, output_dict=True)
    print('Report MLP Classifier ', report_mlp)

    print('Fitting Hierarchical MLP Classifier')
    classifier_mlp.fit(x_train, y_hier_train)
    y_hier_pred_mlp = classifier_mlp.predict(x_test)
    f1_score_mlp = f1(y_true=y_hier_test, y_pred=y_hier_pred_mlp)
    precision_score_mlp = precision(y_true=y_hier_test, y_pred=y_hier_pred_mlp)
    recall_score_mlp = recall(y_true=y_hier_test, y_pred=y_hier_pred_mlp)
    print('F1 Score HiClass MLP', f1_score_mlp, 'Precision Score HiClass MLP', precision_score_mlp, 'Recall Score HiClass MLP', recall_score_mlp)

    hiclass_mlp_scores = {
        'f1': f1_score_mlp,
        'precision': precision_score_mlp,
        'recall': recall_score_mlp
    }
    hi_flat_report = normal_metrics(y_hier_pred_mlp, y_hier_test)

    plot_comparative_graph(report_mlp['weighted avg'], hi_flat_report['weighted avg'], hiclass_mlp_scores)


#main()
#
#lp = LineProfiler()
#lp.add_function(main)
#lp.run('main()')
#lp.print_stats()
if __name__ == '__main__':
    profile_output = "profile_output.prof"
    cProfile.run('main()', filename=profile_output)
