import os
from efficient_net_extract_features import *
from train_classifier import train_classifier_head, train_classifier_sklearn, create_classifier, visualize_results, results_inter_batch, create_random_batch
import math
from sklearn.metrics import accuracy_score
import timm

# EfficientNet-B0-pretrained | EfficientNet-B0-CoralNet-pretrained | Timm-model
NN_NAME = "Timm-model"
# MLPHead | MLPClassifier
CLASSIFIER_NAME = "MLPHead"
BATCH_SIZE = 800
BATCH_SIZE_TEST = 20
NB_ANNOTATIONS = 25

if __name__ == '__main__':
    path_images = './Images'
    path_annotations = './annotations'
    annotations_file = path_annotations + '/annotations.csv'

    test_folder = path_images + '/Test'
    train_folder = path_images + '/Train'

    # load efficientnet
    weights_file = './weights/efficientnet_b0_ver1.pt'
    if NN_NAME == "EfficientNet-B0-pretrained":
        efficientnet = load_model_pretrained()
    elif NN_NAME == "EfficientNet-B0-CoralNet-pretrained":
        efficientnet = load_model(weights_file)
    elif NN_NAME == "Timm-model":
        print("Loading timm model")
        efficientnet = timm.create_model('convnext_base.clip_laion2b', pretrained=True, num_classes=0)
        efficientnet.to(torch.device("cuda:0"))
        efficientnet.eval()

    # list every image in directory
    file_list = os.listdir(path_images)
    images_list = [file for file in file_list if file.endswith(('.JPG', '.jpg'))]

    x_test, y_test, test_batch = create_random_batch(path_images, BATCH_SIZE_TEST, annotations_file, test_folder, efficientnet)

    total_batch = math.ceil((len(images_list) - BATCH_SIZE) / BATCH_SIZE)
    #total_batch = 15
    print(total_batch)
    label_count = total_batch * BATCH_SIZE * NB_ANNOTATIONS
    first_clf = create_classifier(label_count, CLASSIFIER_NAME)  # if label_count > 5000 : clf = 1 layer, clf = 2 layer

    # Metrics in the classifier
    results_per_batch = []

    num_batch = 0
    previous_batch = None
    for num_batch in range(total_batch):
        # make sure GPU is empty
        torch.cuda.empty_cache()
        if CLASSIFIER_NAME == 'MLPHead':
            if previous_batch is not None:
                #test result previous batch = y_predict = first.clf.predict(x_train)
                y_predict_train = first_clf.predict(x_train)
                accuracy = accuracy_score(y_train, y_predict_train)
                #print("Y_train batch_n-1", y_train)
                #print("y_predict_train n-1", y_predict_train)
                print("Accuracy Batch n-1:", accuracy)

        # create new batch
        x_train, y_train, batch = create_random_batch(path_images, BATCH_SIZE, annotations_file, train_folder,
                                                      efficientnet)
        if x_train is None:
            break

        if previous_batch:
            for image_labels in previous_batch.image_labels_list:
                batch.add_image(image_labels)

        print('Size of Batch', batch.label_count())
        if CLASSIFIER_NAME == 'MLPHead':
            y_test, y_pred, loss_curve, train_time = train_classifier_head(x_train, y_train, x_test, y_test, first_clf)
            visualize_results(y_test, y_pred, loss_curve, train_time)
            batch_results = [y_test, y_pred, loss_curve, train_time]

            results_per_batch.append(batch_results)
        elif CLASSIFIER_NAME == 'MLPClassifier':
            y_test, y_pred, loss_curve, train_time = train_classifier_sklearn(x_train, y_train, x_test, y_test, first_clf)
            visualize_results(y_test, y_pred, loss_curve, train_time)
            batch_results = [y_test, y_pred, loss_curve, train_time]
            results_per_batch.append(batch_results)

        previous_batch = batch

    results_inter_batch(results_per_batch)
