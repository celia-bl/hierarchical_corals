from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile
from torch import Tensor

from data_classes import Batch, Patch, Point
import pandas as pd
import torch
import torchvision.transforms as transforms
import params
import csv
import os
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Permet de charger les images tronquées


def crop_patches(im, rowcols, crop_size: int):
    '''
    Crop various sub-images (i.e. patches) from an image
    :param im: Pillow Image
    :param rowcols: list of Points representing the center of the sub-image to crop
    :param crop_size: size of the sub-image you need
    :return: patches : list of Pillow Images
    '''
    patches = []
    try:
        # Définir le padding
        pad = crop_size
        im_padded = ImageOps.expand(im, pad, fill='white')

        # Ajuster les coordonnées des points pour tenir compte du padding
        adjusted_rowcols = [Point(p.x + pad, p.y + pad) for p in rowcols]

        for point in adjusted_rowcols:
            patch = crop_simple(im_padded, point, crop_size)
            patches.append(patch)

    except (OSError, IOError) as e:
        print(f"Error processing image: {e}")
        # Vous pouvez choisir de retourner une liste vide, lever une exception, ou gérer cela autrement
        return []

    return patches


def crop_simple(im: Image, center: Point, crop_size: int) -> Image:
    '''
    Crop a single sub-image given image, center of the point and crop_size
    :param im: Pillow Image
    :param center: Point representing the center of the sub-image to crop
    :param crop_size: size of the sub-image you need
    :return: cropped image
    '''
    print(center)
    upper = int(center.x - crop_size / 2)
    left = int(center.y - crop_size / 2)
    return im.crop((left, upper, left + crop_size, upper + crop_size))


def crop_images(batch: Batch, crop_size: int, path_images_batch:str) -> Batch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device="cpu"
    """

    :param batch : batch with all images to crop
    :param crop_size: size to crop images i.e. input size of efficient net
    :return:
    """
    for image in batch.image_labels_list:
        image_name, rowcols = image.get_points_per_image()
        #image_folder = os.path.join(output_folder, os.path.splitext(image_name)[0])
        #os.makedirs(image_folder, exist_ok=True)
        with Image.open(path_images_batch + '/' + image_name) as im:
            patches = crop_patches(im, rowcols, crop_size)
            #write patches in a folder in order to debug
            for i, patch in enumerate(patches):
                patch_name = f"patch_{i}.jpg"
                #patch_path = os.path.join(image_folder, patch_name)
                #patch.save(patch_path)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
                patch_tensor = transform(patch).to(device) # Normalize tensor
                # Create Patch instance
                patch_point = rowcols[i]
                patch_label = image.get_label_at_point(patch_point)
                new_patch = Patch(patch_point, patch_name, patch_label, patch_tensor)

                image.add_patch(new_patch)

    return batch


def crop_images_in_patches(image_name: str, batch: Batch, crop_size: int, path_images_batch: str) -> tuple[
    Tensor, list[str | None]]:
    """

    :param image_name:
    :param batch:
    :param crop_size:
    :param path_images_batch:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_tensor_list = []
    patch_labels_list = []
    # device="cpu"
    image_label = batch.get_image_labels(image_name)
    image_name, rowcols = image_label.get_points_per_image()
    #image_folder = os.path.join(output_folder, os.path.splitext(image_name)[0])
    #os.makedirs(image_folder, exist_ok=True)

    #patch_folder = 'test_patch'
    #os.makedirs(patch_folder, exist_ok=True)

    with Image.open(path_images_batch + '/' + image_name) as im:
        print("Opening image ", image_name)
        patches = crop_patches(im, rowcols, crop_size)
        print("Cropping patches ", len(patches))
        # write patches in a folder in order to debug
        for i, patch in enumerate(patches):
            #print("PATCH SIZE", patch.size)
            #(224,224)
            patch_name = f"patch_{i}.jpg"
            #patch_path = os.path.join(patch_folder, patch_name)
            #patch.save(patch_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            patch_tensor = transform(patch).to(device)  # Normalize tensor
            #print("PATCH AFTER TRANSFORMATION", patch_tensor.shape)
            #(3,224,224)
            patch_tensor_list.append(patch_tensor)
            # Create Patch instance
            patch_point = rowcols[i]
            patch_label = image_label.get_label_at_point(patch_point)
            patch_labels_list.append(patch_label)
            new_patch = Patch(patch_point, patch_name, patch_label)

            image_label.add_patch(new_patch)

    patches_tensor = torch.stack(patch_tensor_list)
    #print("PATCHES TENSOR SHAPE: ", patches_tensor.shape)
    #(25,3,224,224)
    return patches_tensor, patch_labels_list


def read_csv_file(annotations_file: str, batch: Batch) -> Batch:
    """
    Fill the ImageLabels Object in the batch with the annotations from a csv file
    :param annotations_file: path to csv file
    :param batch: batch with all images name
    :return: list of images and their corresponding annotations 'image_name' : '[(row1, col1, label1), (row2, col2, label2), ...]'
    """
    # 'image_name' : '[(row1, col1, label1), (row2, col2, label2), ...]'
    df = pd.read_csv(annotations_file)

    for index, row in df.iterrows():
        image_name = row['Name']
        row_image = int(row['Row'])
        column_image = int(row['Column'])
        label_image = str(row['Label'])

        # Check if images already in the batch
        #if image_name not in batch:
        #print('No image {}'.format(image_name))
        image_label = batch.get_image_labels(image_name)
        if image_label is not None:
            image_label.add_annotations((row_image, column_image, label_image))

    return batch


def read_txt_file(input_folder, txt_file):
    points_labels = []

    # Combine the input folder and file name to get the full path
    file_path = os.path.join(input_folder, txt_file)

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    continue
                parts = line.split(';')
                row = int(parts[0].strip())
                col = int(parts[1].strip())
                label = parts[2].strip()
                points_labels.append((row, col, label))
    except FileNotFoundError:
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    except Exception as e:
        raise Exception(f"An error occurred while reading the file: {e}")

    return points_labels


def draw_labels_on_image(image_path: str, txt_file_path: str, output_path: str):
    """
    Dessine des points avec des labels et des numéros de patch sur l'image spécifiée.

    :param image_path: Chemin vers l'image sur laquelle dessiner.
    :param txt_file_path: Chemin vers le fichier texte contenant les coordonnées et labels.
    :param output_path: Chemin pour enregistrer l'image avec les labels dessinés.
    """
    # Lire les coordonnées et labels
    points_labels = read_txt_file(os.path.dirname(image_path), os.path.basename(txt_file_path))

    # Ouvrir l'image
    with Image.open(image_path) as im:
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()  # Utiliser une police par défaut pour les labels

        # Dessiner les points, labels et numéros de patch
        for i, (row, col, label) in enumerate(points_labels):
            # Dessiner un point
            draw.ellipse([(col - 5, row - 5), (col + 5, row + 5)], fill='red', outline='red')

            # Dessiner le label et le numéro du patch à une position fixe par rapport au point
            label_offset = 10  # Distance du label par rapport au point
            text_x = col + label_offset
            text_y = row - label_offset
            draw.text((text_x, text_y), f"{label} (#{i})", fill='black', font=font)

        # Assurez-vous que le chemin de sortie se termine par une extension valide
        if not output_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            output_path += '.jpg'  # Ajouter une extension par défaut

        # Enregistrer l'image modifiée
        im.save(output_path)


def get_patch_image_pt(img_name_pt, annotations_path, img_folder):
    base_name, extension_patch_num = img_name_pt.rsplit('.', 1)
    base_name = base_name + '.JPG'
    pt, patch_num = extension_patch_num.rsplit(' ', 1)
    patch_num = int(patch_num)
    return get_patch_image(base_name, patch_num, annotations_path, img_folder)


def get_patch_image(img_name, patch_num, annotations_path, img_folder):
    annotations = pd.read_csv(annotations_path)
    image_annotations = annotations[annotations['Name'] == img_name]
    if len(image_annotations) > patch_num:
        annotation_row = image_annotations.iloc[patch_num]
    else:
        raise IndexError(f"Patch number {patch_num} out of range for image {img_name}")

    img = Image.open(f'{img_folder}/{img_name}')
    row, col = annotation_row['Row'], annotation_row['Column']

    return crop_simple(img, Point(row, col), 224)


def get_patch_ft(feature):
    print('Getting Patch from Feature ... ')
    indices = []
    for file in os.listdir('.features/efficientnet-b0_features'):
        tensor = torch.load(os.path.join('.features/efficientnet-b0_features', file))
        tensor = tensor['features']
        for i in range(tensor.size(0)):
            current_vector = tensor[i].to('cuda')
            if torch.all(current_vector == feature):
                indices.append(i)
                image_name, extension = os.path.splitext(file)
                full_image_name = image_name + '.JPG'
    if len(indices) > 1:
        print(indices, image_name)
        #raise ValueError('More than one feature equal for this image and feature')
        print("feature with more than one match")
        return Image.new('RGB', (224, 224)), 0, (0, 0), 0
    csv_file = params.annotations_file
    rows_columns_labels = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['Name'] == full_image_name:
                rows_columns_labels.append({
                    'row': int(row['Row']),
                    'column': int(row['Column']),
                    'label': row['Label']
                })
    print(full_image_name)
    print(len(rows_columns_labels))
    if len(rows_columns_labels) < 25:
        raise ValueError('Not enough patch in images ?')
    patch_row = rows_columns_labels[indices[0]]['row']
    patch_column = rows_columns_labels[indices[0]]['column']
    patch_label = rows_columns_labels[indices[0]]['label']
    path_img = os.path.join(params.path_images_batches, full_image_name)
    im = Image.open(path_img)
    patch = crop_simple(im, Point(patch_row, patch_column), 224)
    return patch, patch_label, (patch_row, patch_column), full_image_name

