import torch
import os


def find_nearest_images(image_name, directory):
    image_features = torch.load(os.path.join(directory, f"{image_name}.pt"))['features']

    nearest_distances = []
    nearest_image_names = []

    for filename in os.listdir(directory):
        if filename.endswith(".pt") and filename != f"{image_name}.pt":
            current_features = torch.load(os.path.join(directory, filename))['features']

            distance = torch.linalg.norm(image_features - current_features)

            nearest_distances.append(distance)
            nearest_image_names.append(filename[:-3])

    sorted_indices = torch.argsort(torch.tensor(nearest_distances))
    nearest_distances = [nearest_distances[i] for i in sorted_indices]
    nearest_image_names = [nearest_image_names[i] for i in sorted_indices]

    return nearest_image_names, nearest_distances


#nearest_images, distances = find_nearest_images("Garças 06.2018 T2IMG_0641", "./.features/efficientnet-b0_features/")
#print("Images les plus proches :", nearest_images)
#print("Distances correspondantes :", distances)


def extract_images_with_labels(directory, label_shad='SHAD', label_unk='Unk', num_images_per_label=15):
    shad_images = []
    unk_images = []

    # Parcourir tous les fichiers dans le répertoire
    for filename in os.listdir(directory):
        if filename.endswith(".pt"):
            # Charger les données du fichier
            data = torch.load(os.path.join(directory, filename))
            features = data['features']
            labels = data['labels']

            # Vérifier les labels de chaque image
            for i, (feature, label) in enumerate(zip(features, labels)):
                if label == label_shad and len(shad_images) < num_images_per_label:
                    shad_images.append((filename[:-3], i, feature, label))
                elif label == label_unk and len(unk_images) < num_images_per_label:
                    unk_images.append((filename[:-3], i, feature, label))

                if len(shad_images) == num_images_per_label and len(unk_images) == num_images_per_label:
                    break

        if len(shad_images) == num_images_per_label and len(unk_images) == num_images_per_label:
            break

    return shad_images, unk_images


directory = "./.features/efficientnet-b0_features/"
shad_images, unk_images = extract_images_with_labels(directory)

# Affichage des informations des images SHAD
print("Images du label 'SHAD':")
for image_info in shad_images:
    print("Nom de l'image:", image_info[0])
    print("Indice de la feature:", image_info[1])
    print("Label:", image_info[3])
    print()

# Affichage des informations des images Unk
print("Images du label 'Unk':")
for image_info in unk_images:
    print("Nom de l'image:", image_info[0])
    print("Indice de la feature:", image_info[1])
    print("Label:", image_info[3])
    print()