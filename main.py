from extract_features import extract_features, create_full_batch
import torch
import matplotlib.pyplot as plt

NN_NAME_LIST = ['efficientnet-b0']
CROP_SIZE = 224
path_images_batches = './Images-2'
path_annotations = './annotations'
annotations_file = path_annotations + '/annotations.csv'
weights_file = './weights/efficientnet_b0_ver1.pt'

from hiclass import LocalClassifierPerParentNode, LocalClassifierPerNode, LocalClassifierPerLevel
from hierarchical_utils import *
from train_classifier import *
from analyze_data import *
import cProfile


def main(folder_path, flat_classifiers, hierarchical_classifiers, nb_images, intermediate_labels):
    flat_reports = {}
    hi_flat_reports = {}
    hi_reports = {}
    flat_hi_reports = {}
    hi_intermediate_reports = {}
    flat_intermediate_reports = {}
    mispredicted_vectors_flat = {}
    mispredicted_vectors_hi = {}

    #x_train, x_test, y_train, y_test = load_data(folder_path)
    #x_train, x_test, y_train, y_test = load_representative_data(folder_path)
    x_train, x_test, y_train, y_test = load_representative_data_mlc(folder_path)
    #x_train, x_test, y_train, y_test = load_representative_uniform_data(folder_path)
    plot_label_distribution(y_test)

    for classifier_name in flat_classifiers.keys():
        flat_reports[classifier_name] = {}
        flat_hi_reports[classifier_name] = {}
        flat_intermediate_reports[classifier_name] = {}
        mispredicted_vectors_flat[classifier_name] = {}

    for classifier_name in hierarchical_classifiers.keys():
        hi_reports[classifier_name] = {}
        hi_flat_reports[classifier_name] = {}
        hi_intermediate_reports[classifier_name] = {}
        mispredicted_vectors_hi[classifier_name] = {}

    n_runs = 10
    for nb_image in nb_images:
        flat_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}
        flat_hi_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}
        flat_intermediate_run_reports = {classifier_name: [] for classifier_name in flat_classifiers.keys()}

        hi_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}
        hi_flat_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}
        hi_intermediate_run_reports = {classifier_name: [] for classifier_name in hierarchical_classifiers.keys()}

        label_counts = defaultdict(lambda: [])
        for _ in range(n_runs):
            # shuffle training set randomly
            indices = np.random.permutation(range(len(x_train)))

            x_train_shuffled = [x_train[i] for i in indices]
            y_train_shuffled = [y_train[i] for i in indices]
            x_train_subset = x_train_shuffled[:nb_image]
            y_train_subset = y_train_shuffled[:nb_image]


            #load representative data
            #x_train_subset, y_train_subset = load_representative_training_set(x_train, y_train, y_test, nb_image)
            #x_train_subset, y_train_subset = load_uniform_training_data(x_train, y_train, nb_image)

            unique_labels, counts = np.unique(y_train_subset, return_counts=True)
            for label, count in zip(unique_labels, counts):
                label_counts[label].append(count)


            y_hier_train, y_hier_test = convert_labels_to_hierarchy(y_train_subset, y_test)
            y_train_encoded, y_test_encoded, dict_encode = encode_labels(y_train_subset, y_test, 'MLPHead-147')

            for classifier_name, classifier in flat_classifiers.items():
                print(f'Processing {classifier_name}')

                if 'MLPHead' in classifier_name:
                    y_train_subset = y_train_encoded

                flat_report, flat_hi_report, flat_intermediate_metric, mispredicted_vectors_flat_run = train_and_evaluate(
                    classifier, x_train_subset, y_train_subset, x_test, y_test, int_labels=intermediate_labels)
                flat_run_reports[classifier_name].append(flat_report['weighted avg'])
                flat_hi_run_reports[classifier_name].append(flat_hi_report)
                flat_intermediate_run_reports[classifier_name].append(flat_intermediate_metric)

                for mispredicted_feature in mispredicted_vectors_flat_run.keys():
                    if mispredicted_feature not in mispredicted_vectors_flat[classifier_name]:
                        mispredicted_vectors_flat[classifier_name][mispredicted_feature] = {
                            'predicted': [],
                            'true_label': mispredicted_vectors_flat_run[mispredicted_feature]['true_label']
                        }
                    mispredicted_vectors_flat[classifier_name][mispredicted_feature]['predicted'].append(
                        mispredicted_vectors_flat_run[mispredicted_feature]['predicted'])

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

                (hi_report, hi_flat_report,
                 hi_intermediate_metric, mispredicted_vectors_hi_run) = train_and_evaluate_hierarchical(
                    classifier, x_train_subset, x_test, y_hier_train, y_hier_test, int_labels=intermediate_labels)
                hi_run_reports[classifier_name].append(hi_report)
                hi_flat_run_reports[classifier_name].append(hi_flat_report)
                hi_intermediate_run_reports[classifier_name].append(hi_intermediate_metric)

                for mispredicted_feature in mispredicted_vectors_hi_run.keys():
                    if mispredicted_feature not in mispredicted_vectors_hi[classifier_name]:
                        mispredicted_vectors_hi[classifier_name][mispredicted_feature] = {
                            'predicted': [],
                            'true_label': mispredicted_vectors_hi_run[mispredicted_feature]['true_label']
                        }
                    mispredicted_vectors_hi[classifier_name][mispredicted_feature]['predicted'].append(
                        mispredicted_vectors_hi_run[mispredicted_feature]['predicted'])

                print(f'Finish processing {classifier_name}')

        print(flat_run_reports)
        print(flat_hi_run_reports)
        average_label_counts = {label: np.mean(counts) for label, counts in label_counts.items()}
        plot_avg_distrib(average_label_counts)
        # Compute mean and standard deviation for flat classifiers
        for classifier_name in flat_classifiers.keys():
            flat_reports = calculate_means(flat_run_reports, classifier_name, nb_image, flat_reports)
            flat_hi_reports = calculate_means(flat_hi_run_reports, classifier_name, nb_image, flat_hi_reports)
            flat_intermediate_reports = calculate_intermediate_means(flat_intermediate_run_reports, classifier_name, nb_image, flat_intermediate_reports)


        # Compute mean and standard deviation for hierarchical classifiers
        for classifier_name in hierarchical_classifiers.keys():
            hi_reports = calculate_means(hi_run_reports, classifier_name, nb_image, hi_reports)
            hi_flat_reports = calculate_means(hi_flat_run_reports, classifier_name, nb_image, hi_flat_reports)
            hi_intermediate_reports = calculate_intermediate_means(hi_intermediate_run_reports, classifier_name, nb_image, hi_intermediate_reports)

    #print(hi_intermediate_reports)
    #print(flat_intermediate_reports)
    print('Final dict ', flat_reports)
    print('Final dict ', flat_hi_reports)
    print('Final dict ', hi_flat_reports)
    print('Final dict ', hi_reports)
    print('Intermediate dict ', flat_intermediate_reports)

    result_file = './results/test_MLC/'

    save_pickle(flat_reports, result_file + 'flat_reports.pkl')
    save_pickle(flat_hi_reports, result_file + 'flat_hi_reports.pkl')
    save_pickle(hi_flat_reports, result_file + 'hi_flat_reports.pkl')
    save_pickle(hi_reports, result_file + 'hi_reports.pkl')
    save_pickle(flat_intermediate_reports, result_file + 'flat_intermediate_reports.pkl')
    save_pickle(hi_intermediate_reports, result_file + 'hi_intermediate_reports.pkl')
    save_pickle(y_test, result_file + 'y_test.pkl')
    save_pickle(intermediate_labels, result_file + 'intermediate_labels.pkl')

    plot_metrics_by_nb_images(flat_reports, flat_hi_reports, hi_flat_reports, hi_reports)
    plot_intermediate_metrics(flat_intermediate_reports, hi_intermediate_reports)
    comparison_cover(flat_intermediate_reports, hi_intermediate_reports, y_test, intermediate_labels)
    #plot_mispredict_vectors(mispredicted_vectors_hi, mispredicted_vectors_flat, folder_path, annotations_file, path_images_batches)



if __name__ == '__main__':
    flat_classifiers = {
        'MLP': MLPClassifier(hidden_layer_sizes=(200, 100)),
        #'MLPHead': MLPHead(hidden_layers_sizes=(200, 100), output_dim=147)
    }

    hierarchical_classifiers = {
        #'LCPPN-1': MLPHead(hidden_layers_sizes=(100,), output_dim=147)
        'LCPPN-1': MLPClassifier(hidden_layer_sizes=(200, 100))
        #'LCPPN-2': MLPHead(hidden_layers=(100,))
        #'LCPN-1': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }
    nb_images = [200, 600, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 8000, 10000, 20000, 30000, 405800]
    #nb_images = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400]
    #nb_images = [20, 50]
    folder_path = '.features/efficientnet-b0_features'
    #intermediate_labels = ['Bleached', 'Corals', 'Algae', 'Non Calcified', 'Calcified', 'Hard', 'Soft', 'Substrates', 'Rock', 'SHAD', 'Unk']
    intermediate_labels = ['Corals', 'Algae', 'Hard', 'Soft', 'Sand']
    main(folder_path, flat_classifiers, hierarchical_classifiers, nb_images, intermediate_labels)
