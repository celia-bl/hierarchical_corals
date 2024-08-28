from data_classes import Batch, ImageLabels
import os
from img_transformation import read_csv_file, read_txt_file
from typing import Optional
from feature_extractor import FeatureExtractor
import shutil
import params as p


def clean_folders(folder_path: str):
    clean_test_folder(folder_path)

    print('Images in test moved and folder deleted')
    print('Removing everything in features folder')
    features_folder = p.features_folder
    for filename in os.listdir(features_folder):
        file_path = os.path.join(features_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Error in suppressing {file_path}: {e}")


def clean_test_folder(path_img):
    test_folder = path_img + '/Test'
    if os.path.exists(test_folder):
        image_files = [f for f in os.listdir(test_folder) if os.path.isfile(os.path.join(test_folder, f))]

        destination_folder = path_img
        for image_file in image_files:
            source_path = os.path.join(test_folder, image_file)
            destination_path = os.path.join(destination_folder, image_file)
            shutil.move(source_path, destination_path)

        shutil.rmtree(test_folder)

    print('Images in test moved and folder deleted')


def create_full_batch(input_folder: str, annotations_file: str):
    batch = Batch()
    files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]

    for file_name in files:
        image_label = ImageLabels(file_name)
        batch.add_image(image_label)

    batch = read_csv_file(annotations_file, batch)

    return batch


def process_single_image(input_folder):
    batch = Batch()
    files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]

    if not files:
        return batch  # Return an empty batch if no images are found

    # Sélectionner la première image trouvée
    file_name = files[0]
    image_label = ImageLabels(file_name)
    batch.add_image(image_label)

    # Lire les annotations pour cette image
    points_labels = read_txt_file(input_folder, file_name + '.txt')
    for annotation in points_labels:
        image_label.add_annotations(annotation)

    return batch


def create_full_batch_txt(input_folder: str):
    batch = Batch()
    files = [file for file in os.listdir(input_folder) if file.endswith(('.JPG', '.jpg'))]

    for file_name in files:
        image_label = ImageLabels(file_name)
        batch.add_image(image_label)
        points_labels = read_txt_file(input_folder, file_name+'.txt')
        for annotation in points_labels:
            image_label.add_annotations(annotation)

    return batch


def extract_features(batch, crop_size, input_folder, model_name, weights_file: Optional):
    batch_size = batch.batch_size()
    if model_name == 'efficientnet-b0':
        feature_extractor = FeatureExtractor(model_name, weights_file=weights_file)
    else:
        feature_extractor = FeatureExtractor(model_name)

    features, labels = feature_extractor.get_features(batch, crop_size, input_folder)

    return features, labels


def extract_all_features():
    NN_NAME_LIST = p.NN_NAME_LIST
    CROP_SIZE = p.CROP_SIZE
    path_images_batches = p.path_images_batches
    annotations_file = p.annotations_file
    weights_file = p.weights_file

    full_batch = create_full_batch(path_images_batches, annotations_file)
    for j in range(len(NN_NAME_LIST)):
        extract_features(full_batch, CROP_SIZE, path_images_batches, NN_NAME_LIST[j], weights_file)


def sort_images_by_year(images_path: str):
    years = p.YEAR_LIST
    if not any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.JPG'] for filename in os.listdir(images_path)):
        print('No images found with extension .jpg, .JPG, .jpeg, .png')
        return
    for filename in os.listdir(images_path):
        filepath = os.path.join(images_path, filename)

        if os.path.isfile(filepath) and any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.JPG']):
            #print('Processing ' + filename)
            found_year = None
            for year in years:
                if year in filename:
                    found_year = year
                    break

            if found_year:
                #print('Found year ' + found_year)
                year_folder = os.path.join(images_path, found_year)
                if not os.path.exists(year_folder):
                    print('Creating folder ' + year_folder)
                    os.makedirs(year_folder)
                print('Moving ' + filepath + ' to ' + year_folder)
                shutil.move(filepath, os.path.join(year_folder, filename))


def de_sort_images(images_path: str):
    print('De-sorting images...')
    years = p.YEAR_LIST
    for year in years:
        year_folder = os.path.join(images_path, year)
        if os.path.isdir(year_folder):
            clean_test_folder(year_folder)
            for filename in os.listdir(year_folder):
                filepath = os.path.join(year_folder, filename)
                if os.path.isfile(filepath) and any(
                        filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.JPG']):
                    #print('Moving ' + filepath + ' to ' + images_path)
                    shutil.move(filepath, os.path.join(images_path, filename))
            shutil.rmtree(year_folder)
            print('Removed folder ', year_folder)
