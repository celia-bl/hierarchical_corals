import params as p
from analyze_data import (analyze_labels,
                          compute_probabilities_labels,
                          compute_probabilities_patch)
from extract_features import clean_test_folder, sort_images_by_year, create_full_batch, de_sort_images


if __name__ == '__main__':
    path_img = p.path_images_batches
    de_sort_images(path_img)
    full_batch = create_full_batch(path_img, p.annotations_file)
    analyze_labels(full_batch, 'TOTAL DATASET')
    compute_probabilities_labels(full_batch)
    compute_probabilities_patch(full_batch)
    clean_test_folder(p.path_images_batches)
    sort_images_by_year(p.path_images_batches)

    for i in range(len(p.YEAR_LIST)):
        path_img = p.path_images_batches + '/' + p.YEAR_LIST[i]
        clean_test_folder(path_img)
        full_year_batch = create_full_batch(path_img, p.annotations_file)

        analyze_labels(full_year_batch, 'TOTAL DATASET ' + p.YEAR_LIST[i])
        compute_probabilities_labels(full_year_batch)
        compute_probabilities_patch(full_year_batch)

    de_sort_images(p.path_images_batches)
