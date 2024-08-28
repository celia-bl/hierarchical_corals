from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from hiclass.metrics import f1, precision, recall
import hierarchy_dictionary as hi_dic
from plot_logic import *
from calcul_logic import *
from hiclass.HierarchicalClassifier import make_leveled


def convert_labels_to_hierarchy(y_train, y_test):
    y_hier_train = np.array([hi_dic.find_hierarchy_path(label) for label in y_train], dtype=object)
    y_hier_test = np.array([hi_dic.find_hierarchy_path(label) for label in y_test], dtype=object)
    return y_hier_train, y_hier_test


def convert_label_to_hierarchy(y_train):
    y_hier_train = np.array([hi_dic.find_hierarchy_path(label) for label in y_train], dtype=object)
    return y_hier_train


def convert_hierarchy_to_label(y_hier_train):
    while y_hier_train[-1] == ',' or y_hier_train[-1] == ' ' or y_hier_train[-1] == '' or y_hier_train[-1] == "":
        y_hier_train = y_hier_train[:-1]
    y_train = y_hier_train[-1]
    return y_train


def extract_last_element(list_of_lists):
    return [sublist[-1] for sublist in list_of_lists]


def hierarchical_metrics_from_flat(y_pred, y_true):
    y_pred_hierarchized, y_true_hierarchized = convert_labels_to_hierarchy(y_pred, y_true)
    flat_hi_report = get_hierarchical_scores_from_hiclass(y_pred_hierarchized, y_true_hierarchized)
    return flat_hi_report


def normal_metrics(y_hier_pred, y_hier_true):
    new_y_pred = []
    for list_label in y_hier_pred :
        new_list_label = convert_hierarchy_to_label(list_label)
        print(list_label, new_list_label)
        new_y_pred.append(new_list_label.tolist())

    #print(new_y_pred)
    #print(len(new_y_pred))
    #new_y_pred = np.array(new_y_pred, dtype=object)
    y_true_last = extract_last_element(y_hier_true)
    #print(y_true_last)
    #print(len(y_true_last))
    # Combine y_true_last and y_pred_last to get all unique classes
    all_classes = set(y_true_last + new_y_pred)
    #print(len(all_classes))

    # Sort the classes alphabetically
    class_names = sorted(all_classes)

    # Calculate accuracy
    accuracy = accuracy_score(y_true_last, new_y_pred)
    print("Accuracy:", accuracy)

    # Generate classification report
    report = classification_report(y_true_last, new_y_pred, target_names=class_names, zero_division=0, output_dict=True)

    print("Classification Report:\n", report)

    #plt.figure(figsize=(15, 15))
    #ConfusionMatrixDisplay.from_predictions(y_true_last, y_pred_last, ax=plt.gca(), xticks_rotation=90)

    #plt.show()
    return report


def decode_labels(encoded_labels, label_to_index):
    if label_to_index is None:
        decoded_labels = encoded_labels
    else:
        index_to_label = {index: label for label, index in label_to_index.items()}
        decoded_labels = [index_to_label.get(index, "ERR") for index in encoded_labels]
    return decoded_labels


def get_hierarchical_scores_from_hiclass(y_hier_pred, y_hier_true):
    f1_score_val = f1(y_hier_true, y_hier_pred)
    precision_score_val = precision(y_hier_true, y_hier_pred)
    recall_score_val = recall(y_hier_true, y_hier_pred)
    hi_report = {
        'precision': precision_score_val,
        'recall': recall_score_val,
        'f1-score': f1_score_val,
        'support': len(y_hier_pred)
    }
    return hi_report


def metrics_intermediate_labels(y_true, y_pred, y_train, intermediate_labels):
    intermediate_metrics = {}

    for label in intermediate_labels:
        intermediate_metrics[label] = {}
        intermediate_metrics[label]['y_true'] = []
        intermediate_metrics[label]['y_pred'] = []

    for true_labels, pred_labels in zip(y_true, y_pred):
        for label in intermediate_labels:
            if label in true_labels or label in pred_labels:
                intermediate_metrics[label]['y_true'].append(label in true_labels)
                intermediate_metrics[label]['y_pred'].append(label in pred_labels)

    for label in intermediate_labels:
        y_true_intermediate = intermediate_metrics[label]['y_true']
        y_pred_intermediate = intermediate_metrics[label]['y_pred']

        intermediate_metrics[label]['accuracy'] = accuracy_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['recall'] = recall_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['precision'] = precision_score(y_true_intermediate, y_pred_intermediate, zero_division=0)
        intermediate_metrics[label]['f1_score'] = f1_score(y_true_intermediate, y_pred_intermediate)
        intermediate_metrics[label]['test_support'] = y_true_intermediate.count(True)
        intermediate_metrics[label]['test_pred'] = count_label_occurrences(y_pred, label)
        intermediate_metrics[label]['train_support'] = count_label_occurrences(y_train, label)

        del intermediate_metrics[label]['y_true']
        del intermediate_metrics[label]['y_pred']

    return intermediate_metrics


def train_and_evaluate_hierarchical(classifier, x_train, x_test, y_hier_train, y_hier_test, int_labels=None):
    classifier.fit(x_train, y_hier_train)
    y_hier_pred = classifier.predict(x_test)

    # normal scores from normal metrics with weighted avg
    hi_flat_report = normal_metrics(y_hier_pred, y_hier_test)['weighted avg']
    hi_report = get_hierarchical_scores_from_hiclass(y_hier_pred, y_hier_test)

    mispredicted_vectors = {}
    y_hier_test_hi = make_leveled(y_hier_test)
    for i, (pred, true) in enumerate(zip(y_hier_pred, y_hier_test_hi)):
        if pred[-1] != true[-1]:
            mispredicted_vectors[tuple(x_test[i])] = {
                'predicted': tuple(pred),
                'true_label': true
            }

    if int_labels is not None:

        intermediate_metrics = metrics_intermediate_labels(y_hier_test, y_hier_pred, y_hier_train, int_labels)
    else:
        intermediate_metrics = None
    return hi_report, hi_flat_report, intermediate_metrics, mispredicted_vectors



