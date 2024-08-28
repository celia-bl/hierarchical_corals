from extract_features import *
import torch
from img_transformation import draw_labels_on_image
MLC_folder = 'MLC/2010'
weights_file = './weights/efficientnet_b0_ver1.pt'

"""
batch = create_full_batch_txt(MLC_folder)
#batch = process_single_image(MLC_folder)
extract_features(batch, 224, MLC_folder, 'efficientnet-b0', weights_file)
"""


"""
img = MLC_folder + "/mcr_lter3_fringingreef_pole5-6_qu6_20100413.jpg"
txt = img + '.txt'
output = MLC_folder + '/img.png'
draw_labels_on_image(img, txt, output)
"""


print(torch.load('.features/efficientnet-b0_features/mcr_lter2_out17m_pole1-2_qu4_20080410.pt'))

"""
image_path = MLC_folder + '/mcr_lter2_out17m_pole1-2_qu4_20080410.jpg'
txt_path = image_path + '.txt'
output_path = './test_patch/img.jpg'
draw_labels_on_image(image_path, txt_path, output_path)
"""

