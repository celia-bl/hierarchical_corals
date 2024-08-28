import pandas as pd
from data_classes import Point
from img_transformation import crop_simple
from PIL import Image

annotations_file = 'annotations/annotations.csv'

df = pd.read_csv(annotations_file)
points = []
labels = []
for index, row in df.iterrows():
    image_name = row['Name']
    row_image = int(row['Row'])
    column_image = int(row['Column'])
    label_image = str(row['Label'])

    if image_name == 'Gaçras 11.2021 T7 DSCN0046.JPG':
        labels.append(label_image)
        points.append(Point(row_image, column_image))

im = Image.open('Images-2/Gaçras 11.2021 T7 DSCN0046.JPG')
for i, point in enumerate(points):
    patch = crop_simple(im, point, 224)
    patch_path='patch_im_article/patch_' + str(i) + '.jpg'
    patch.save(patch_path)