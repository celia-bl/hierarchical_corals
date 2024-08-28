import torch.cuda
import params as p
from feature_extractor import FeatureExtractor
from sklearn.model_selection import train_test_split
import numpy as np
from train_classifier import (train_classifier_head,
                              create_classifier)
from analyze_data import (visualize_results,
                          complete_analyze,
                          analyze_probas,
                          plot_report,
                          plot_usual_errors)
from extract_features import clean_test_folder, de_sort_images

if __name__ == '__main__':
    for i in range(len(p.YEAR_LIST)):
        # TODO : make curve with accuracy + year by year + 2 years by 2 years (2018 + 2019, 2020 + 2021, 2022)
        accuracies = {network: {classifier: [] for classifier in p.CLASSIFIER_NAME_LIST} for network in set(p.NN_NAME_LIST)}
        times = {network: {classifier: [] for classifier in p.CLASSIFIER_NAME_LIST} for network in set(p.NN_NAME_LIST)}
        for j in range(len(p.NN_NAME_LIST)):
            if p.NN_NAME_LIST[j] == 'efficientnet-b0':
                feature_extractor = FeatureExtractor(p.NN_NAME_LIST[j], p.weights_file)
            else:
                feature_extractor = FeatureExtractor(p.NN_NAME_LIST[j])

            features, labels = FeatureExtractor.get_features(feature_extractor)
            print(features.shape, len(labels))
            unique_labels, counts = np.unique(labels, return_counts=True)
            valid_labels = unique_labels[counts >= 2]
            print(len(set(labels)), len(set(valid_labels)))
            mask_valid = torch.tensor([label in valid_labels for label in labels], dtype=torch.bool)

            features_filtered = features[mask_valid]
            labels_filtered = [labels[i] for i in range(len(labels)) if labels[i] in valid_labels]
            print(features_filtered.shape, len(labels_filtered))
            (feat_train, feat_test,
             label_train, label_test) = train_test_split(features_filtered,
                                                         labels_filtered, test_size=0.1,
                                                         random_state=41, stratify=labels_filtered)
            # TODO : make 10 samples of each accuracies to have avg + std
            for k in range(len(p.CLASSIFIER_NAME_LIST)):
                clf = create_classifier(len(set(labels)), p.CLASSIFIER_NAME_LIST[k])
                (y_test, y_pred, score,
                 loss_curve, train_time,
                 probas, dict_dec) = train_classifier_head(feat_train, label_train,
                                                           feat_test, label_test,
                                                           clf, p.CLASSIFIER_NAME_LIST[k])
                accuracy, report = visualize_results(y_test, y_pred, loss_curve, train_time)
                accuracies[p.NN_NAME_LIST[j]][p.CLASSIFIER_NAME_LIST[k]].append(accuracy)
                times[p.NN_NAME_LIST[j]][p.CLASSIFIER_NAME_LIST[k]].append(train_time)
                complete_analyze(label_train, label_test, report, y_pred, score)
                plot_report(report)
                analyze_probas(probas, dict_dec, y_test, feat_test, y_pred)
