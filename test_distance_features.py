from hierarchical_utils import *

if __name__ == '__main__':
    path_images = './Images-2'
    path_annotations = './annotations'
    annotations_file = path_annotations + '/annotations.csv'
    folder_path = '.features/efficientnet-b0_features'
    target_labels = ['Unk', 'SHAD', 'TFL', 'Turf_sand', 'PAL', 'ZOS', 'Rock', 'BL', 'SINV_BLC']
    for label in target_labels:
        image_names, features, labels = get_all_features_img_name(folder_path)
        normal_dict, pca_dict = calculate_distance_features(image_names, features, labels, label)
        print(normal_dict)
        plot_distant_features(normal_dict, annotations_file, path_images, label, "NORMAL")
        plot_distant_features(pca_dict, annotations_file, path_images, label, "PCA")
