import torch.cuda
import params as p
from feature_extractor import FeatureExtractor
from train_classifier import (create_random_batch,
                              train_classifier_head,
                              create_classifier)
from analyze_data import (visualize_final,
                          visualize_results,
                          analyze_labels,
                          complete_analyze,
                          compute_probabilities_labels,
                          analyze_probas,
                          compute_probabilities_patch)
from extract_features import clean_test_folder


if __name__ == '__main__':
    accuracies = {network: {classifier: [] for classifier in p.CLASSIFIER_NAME_LIST} for network in set(p.NN_NAME_LIST)}
    times = {network: {classifier: [] for classifier in p.CLASSIFIER_NAME_LIST} for network in set(p.NN_NAME_LIST)}
    clean_test_folder(p.path_images_batches)
    # make batch test
    test_batch = create_random_batch(p.path_images_batches, p.BATCH_SIZE_TEST, p.annotations_file, 'TEST')
    #analyze_labels(test_batch, 'TEST')
    for i in range(len(p.BATCH_SIZE_LIST)):
        train_batch = create_random_batch(p.path_images_batches, p.BATCH_SIZE_LIST[i], p.annotations_file, 'TRAIN')
        #analyze_labels(train_batch, 'TRAIN')
        for j in range(len(p.NN_NAME_LIST)):
            if p.NN_NAME_LIST[j] == 'efficientnet-b0':
                feature_extractor = FeatureExtractor(p.NN_NAME_LIST[j], p.BATCH_SIZE_LIST[i], p.weights_file)
            else:
                feature_extractor = FeatureExtractor(p.NN_NAME_LIST[j], p.BATCH_SIZE_LIST[i])

            test_features, test_labels = feature_extractor.get_features(test_batch, p.CROP_SIZE, p.path_images_batches)

            features, labels = feature_extractor.get_features(train_batch, p.CROP_SIZE, p.path_images_batches)
            #print("SHAPE", features.shape, "\n", len(labels), "\n")
            #print(labels)
            label_count = p.BATCH_SIZE_LIST[i] * p.NB_ANNOTATIONS
            #print("TEST FEATURES", test_features.shape, "len test labels", len(test_labels))

            for k in range(len(p.CLASSIFIER_NAME_LIST)):
                # if label_count > 5000 : clf = 1 layer, clf = 2 layer
                clf = create_classifier(label_count,
                                        p.CLASSIFIER_NAME_LIST[k])
                (y_test, y_pred, score,
                 loss_curve, train_time,
                 probas, dict_dec) = train_classifier_head(features, labels,
                                                           test_features, test_labels,
                                                           clf, p.CLASSIFIER_NAME_LIST[k])

                accuracy, report = visualize_results(y_test, y_pred, loss_curve, train_time)
                #analyze_probas(probas, dict_dec, y_test)
                accuracies[p.NN_NAME_LIST[j]][p.CLASSIFIER_NAME_LIST[k]].append(accuracy)
                times[p.NN_NAME_LIST[j]][p.CLASSIFIER_NAME_LIST[k]].append(train_time)
                complete_analyze(train_batch, test_batch, report, y_pred, score)

            torch.cuda.empty_cache()
            del features, labels
        if p.BATCH_SIZE_LIST[i] == 806:
            #analyze_labels(train_batch+test_batch, 'TOTAL DATASET')
            #compute_probabilities_labels(train_batch+test_batch)
            compute_probabilities_patch(train_batch+test_batch)
    #visualize_final(accuracies, times, p.BATCH_SIZE_LIST)
