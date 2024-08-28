from sklearn.neural_network import MLPClassifier
from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode, LocalClassifierPerLevel
from hierarchical_utils import *
from mlp_head import MLPHead
from train_classifier import encode_labels
from extract_features import extract_all_features
import cProfile
from collections import Counter


def main(folder_path, flat_classifiers, hierarchical_classifiers, nb_images, intermediate_labels):
    flat_reports = {}
    hi_flat_reports = {}
    hi_reports = {}
    flat_hi_reports = {}
    hi_intermediate_reports = {}
    flat_intermediate_reports = {}

    x_train, x_test, y_train, y_test = load_data(folder_path)

    for classifier_name in flat_classifiers.keys():
        flat_reports[classifier_name] = {}
        flat_hi_reports[classifier_name] = {}
        flat_intermediate_reports[classifier_name] = {}

    for classifier_name in hierarchical_classifiers.keys():
        hi_reports[classifier_name] = {}
        hi_flat_reports[classifier_name] = {}
        hi_intermediate_reports[classifier_name] = {}

    for nb_image in nb_images:

        x_train = x_train[:nb_image * 25]
        y_train = y_train[:nb_image * 25]
        y_hier_train, y_hier_test = convert_labels_to_hierarchy(y_train, y_test)

        #x_train = torch.tensor(x_train, dtype=torch.float32)
        #x_test = torch.tensor(x_test, dtype=torch.float32)

        y_train_encoded, y_test_encoded, dict_encode = encode_labels(y_train, y_test, 'MLPHead-147')

        for classifier_name, classifier in flat_classifiers.items():
            print(f'Processing {classifier_name}')

            if 'MLPHead' in classifier_name:
                y_train = y_train_encoded

            flat_report, flat_hi_report, flat_intermediate_metric = train_and_evaluate(classifier, x_train, y_train, x_test, y_test, int_labels=intermediate_labels)
            flat_reports[classifier_name][nb_image] = flat_report['weighted avg']
            flat_hi_reports[classifier_name][nb_image] = flat_hi_report
            flat_intermediate_reports[classifier_name][nb_image] = flat_intermediate_metric

            print(f'Finish processing {classifier_name}')

        for classifier_name, node_classifier in hierarchical_classifiers.items():

            print(f'Processing {classifier_name}')

            if 'LCPPN' in classifier_name:
                classifier = LocalClassifierPerParentNode(local_classifier=node_classifier)
            elif 'LCPN' in classifier_name:
                classifier = LocalClassifierPerNode(local_classifier=node_classifier)
            elif 'LCPL' in classifier_name:
                classifier = LocalClassifierPerLevel(local_classifier=node_classifier)
            else:
                raise ValueError(f'Unknown classifier {classifier_name}, specify LCPPN or LCPN')
            if isinstance(node_classifier, MLPHead):
                print('Yes ')
            hi_report, hi_flat_report, hi_intermediate_metric = train_and_evaluate_hierarchical(classifier, x_train, x_test, y_hier_train, y_hier_test, int_labels=intermediate_labels)
            hi_reports[classifier_name][nb_image] = hi_report
            hi_flat_reports[classifier_name][nb_image] = hi_flat_report
            hi_intermediate_reports[classifier_name][nb_image] = hi_intermediate_metric

            print(f'Finish processing {classifier_name}')

    print(hi_intermediate_reports)
    print(flat_intermediate_reports)
    plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports)
    plot_intermediate_metrics(flat_intermediate_reports, hi_intermediate_reports)
    comparison_cover(flat_intermediate_reports, hi_intermediate_reports, y_test, intermediate_labels)
    #plot_comparative_graph(flat_reports, hi_reports, hi_flat_reports, list(flat_classifiers.keys()), list(hierarchical_classifiers.keys()))


if __name__ == '__main__':
    flat_classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(200, 100)),
        #'MLPHead': MLPHead(hidden_layers_sizes=(200, 100), output_dim=147)
    }

    hierarchical_classifiers = {
        #'LCPPN-1': MLPHead(hidden_layers_sizes=(100,), output_dim=147)
        'LCPPN-1': MLPClassifier(hidden_layer_sizes=(200,100))
        #'LCPPN-2': MLPHead(hidden_layers=(100,))
        #'LCPN-1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }
    #nb_images = [100, 200, 400, 800, 856]
    nb_images = [100]
    folder_path = '.features/efficientnet-b0_features'
    intermediate_labels = ['Bleached', 'Corals', 'Algae', 'Non Calcified', 'Calcified', 'Hard', 'Soft', 'Substrates', 'Rock', 'SHAD', 'Unk']
    main(folder_path, flat_classifiers, hierarchical_classifiers, nb_images, intermediate_labels)
