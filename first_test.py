# import every files
# donne le chemin des images
# img_transformation
# efficient_net_extract_features
# train_classifier
# print result
import torch

from img_transformation import crop_images_in_patches, read_csv_file
import os
from data_classes import *
from efficient_net_extract_features import *
from train_classifier import train_classifier, create_classifier, visualize_results, results_inter_batch, create_random_batch
import torch.nn.functional as F
import math
import pickle
import copy
# from train_classifier import *

# EfficientNet-B0-pretrained | EfficientNet-B0-CoralNet-pretrained
NN_NAME = "EfficientNet-B0-CoralNet-pretrained"
BATCH_SIZE = 20
NB_ANNOTATIONS = 25

if __name__ == '__main__':
    path_images = './Images'
    path_annotations = './annotations'
    annotations_file = path_annotations + 'annotations.csv'

    test_folder = path_images + '/Test'
    train_folder = path_images + '/Train'

    # load efficientnet
    weights_file = './weights/efficientnet_b0_ver1.pt'
    if NN_NAME == "EfficientNet-B0-pretrained":
        efficientnet = load_model_pretrained()
    elif NN_NAME == "EfficientNet-B0-CoralNet-pretrained":
        efficientnet = load_model(weights_file)

    # list every image in directory
    file_list = os.listdir(path_images)
    images_list = [file for file in file_list if file.endswith(('.JPG', '.jpg'))]

    x_test, y_test, test_batch = create_random_batch(path_images, int(0.2*BATCH_SIZE), annotations_file, test_folder, efficientnet)

    total_batch = math.ceil(len(images_list) - BATCH_SIZE / BATCH_SIZE)
    label_count = total_batch * BATCH_SIZE * NB_ANNOTATIONS
    first_clf = create_classifier(label_count)  # if label_count > 5000 : clf = 1 layer, clf = 2 layer

    # Metrics in the classifier
    results_per_batch = []

    num_batch = 0
    previous_batch = None
    for num_batch in range(total_batch-1):
        # Test classifier
        #if num_batch > 0 :
        #    for param_group in first_clf.optimizer.param_groups:
        #        print("Parameters of this group before:")
        #        for key, value in param_group.items():
        #            print(f"{key}: {value}")
        #        print("\n")
            #lr_debut = first_clf.optimizer.param_groups[0]['lr']
            #print("Learning rate début", str(lr_debut))
        #    classifier_before_training = copy.deepcopy(first_clf.model.state_dict())
        # Stock patches
        #output_folder = os.path.join(path_images, 'patches')
        #os.makedirs(output_folder, exist_ok=True)

        # make sure GPU is empty
        torch.cuda.empty_cache()

        # create new batch
        x_train, y_train, batch = create_random_batch(path_images, BATCH_SIZE, annotations_file, train_folder,
                                                      efficientnet)

        # take images in the list (can also make to take images randomly)
        for image in images_list[num_batch*BATCH_SIZE:num_batch+BATCH_SIZE-1]:
            image_label = ImageLabels(image)
            batch.add_image(image_label)

        # save annotations and images from a batch into a Batch object
        batch = read_csv_file(annotations_file, batch)
        #print('Batch size ', batch.batch_size())
        #print('Label count ', batch.label_count())

        for image_name in images_list:
            print("Image name: ", image_name)
            if NN_NAME == "EfficientNet-B0-pretrained":
                with torch.no_grad():
                    unpooled_features = efficientnet.features(crop_images_in_patches(image_name, batch, 224, path_images))
                pooled_features = F.avg_pool2d(unpooled_features, kernel_size=7, stride=1)
                features = pooled_features.view(unpooled_features.size(0), -1)
            elif NN_NAME == "EfficientNet-B0-CoralNet-pretrained":
                with torch.no_grad():
                    # list_patch_stacked = crop_images_in_patches(image_name, batch, 224, path_images_batch)
                    # features = efficientnet(list_patch_stacked)
                    # del list_patch_stacked
                    features = efficientnet(crop_images_in_patches(image_name, batch, 224, path_images))

            # add every feature to every patch
            patches = batch.get_image_labels(image_name).get_list_patches()
            for j, patch in enumerate(patches):
                patch.add_features(features[j])
            del features
            torch.cuda.empty_cache()

        if previous_batch:
            for image_labels in previous_batch.image_labels_list:
                batch.add_image(image_labels)

        print('Start training classifier Batch ... '+ str(num_batch))
        print('Size of Batch', batch.label_count())
        y_test, y_pred, loss_curve, train_time = train_classifier(batch, first_clf)
        visualize_results(y_test, y_pred, loss_curve, train_time)
        batch_results = [y_test, y_pred, loss_curve, train_time]

        #classifier_after_training = copy.deepcopy(first_clf.model.state_dict())
        #print("Poids du classifier avant l'entraînement du batch :")
        #if num_batch > 0 :
        #    print(classifier_before_training)
        #print("Poids du classifier après l'entraînement du batch :")
        #print(classifier_after_training)
        #lr_fin = first_clf.optimizer.param_groups[0]['lr']
        #print('Learning rate nfin', str(lr_fin))
        #for param_group in first_clf.optimizer.param_groups:
        #    print("Parameters of this group after:")
        #    for key, value in param_group.items():
        #        print(f"{key}: {value}")
        #    print("\n")
        results_per_batch.append(batch_results)
        previous_batch = batch

    results_inter_batch(results_per_batch)