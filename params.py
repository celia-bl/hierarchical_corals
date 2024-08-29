import pandas as pd
from utils import CaseInsensitiveDict
# models in timm.list_models() 'efficientnet_b0.ra_in1k', 'convnext_base.fb_in22k', 'convnext_base.clip_laion2b',
#NN_NAME_LIST = [ 'efficientnet_b5.sw_in12k_ft_in1k', 'convnext_base.clip_laion2b', 'efficientnet_b0.ra_in1k', 'convnext_base.fb_in22k', 'efficientnet-b0']
NN_NAME_LIST = ['efficientnet-b0']
# MLPHead | MLPClassifier | MLPHead-resized
#CLASSIFIER_NAME_LIST = ['MLPHead-147', 'MLPHead-40']
CLASSIFIER_NAME_LIST = ['MLPHead-147']
#YEAR_LIST = ['2018', '2019', '2020', '2021', '2022']
YEAR_LIST = ['2020']
#BATCH_SIZE_LIST = [806, 700, 600, 500, 400, 300, 200, 100, 50, 20]
BATCH_SIZE_LIST = [806]
BATCH_SIZE_TEST = 50
#BATCH_SIZE_TEST=5
NB_ANNOTATIONS = 25
CROP_SIZE = 224
path_images_batches = './Images-2'
path_annotations = './annotations'
annotations_file = path_annotations + '/annotations.csv'
weights_file = './weights/efficientnet_b0_ver1.pt'
features_folder = './.features'
test_folder = path_images_batches + '/Test'

labelset_file = "labelset/labelset.csv"
labelset_df = pd.read_csv(labelset_file)

# Dict between labels and label id
label_to_id = {label: label_id for label_id, label in zip(labelset_df["Label ID"], labelset_df["Short Code"])}
label_to_line_number = {label: index for index, label in enumerate(labelset_df["Short Code"])}
'''
label_to_line_number = CaseInsensitiveDict({
    'CCA':0,
    'Porit':1,
    'Macro':2,
    'Sand':3,
    'Off':4,
    'Turf':5,
    'Monti':6,
    'Acrop':7,
    'Pavon':8,
    'Pocill':9,
    'Monta':10,
    'Lepto':11,
    'Lobo':12,
    'Pachy':13,
    'Acan':14,
    'Astreo':15,
    'Cypha':16,
    'Favia':17,
    'Fung':18,
    'Gardin':19,
    'Herpo':20,
    'Psam':21,
    'Sando':22,
    'Stylo':23,
    'Tuba':24,
    'Soft':25,
    'Mille':26,
    'Lepta':27,
    'P. Rus':28,
    'P mass':29,
    'P. Irr':30
})
'''

short_codes_unique = labelset_df["Short Code"].unique()
index_list = list(range(len(short_codes_unique)))
line_number_to_label = {v: k for k, v in label_to_line_number.items()}

code_label_full_name = {
    'PAL':'Palythoa spp.',
    'Sand':'Sand',
    'Turf_sand':'Turf and sand',
    'TFL':'Turf Filamenteous',
    'CNA/CC':'Calcifying calcareous crustose algae: DHC',
    'DICT4':'Dictyota spp.',
    'ZOS': 'Zoanthus Sociatus',
    'TCC': 'Calcareous Turf',
    'SIDS': 'Siderastrea Stellata',
    'DICMER':'Dictyota Mertensii',
    'SHAD':'Shadow',
    'DICTC':'Dictyota Ciliolata',
    'CIAN4':'Cyanobacteria films',
    'Unk':'Unknown',
    'SINV_BLC':'Soft Coral Bleached',
    'POA':'Porites Astreoides',
    'BL': 'Bleached Coral Point',
    'Rock':'Rock',
    'DICPD':'Dictyopteris Delicatula',
    'DICP':'Dictyopteris spp.',
    'LAU':'Laurencia spp.',
    'WRAN':'Wrangelia',
    'ARC': 'Substrate: Unconsolidated (soft)',
    'MIL':'Millepora spp.',
    'FAV':'Favia Gravida',
    'CIAN2':'Cyanobacteria',
    'ZOA':'Zoanthus spp.',
    'Gespp.':'Gelidiella spp.',
    'MIA': 'Millepora Alcicornis',
    'DICPP':'Dictyopteris Plagiogramma',
    'RD_HC': 'Recent Dead Coral',
    'SARG':'Sargassum',
    'PRPV':'Protopalythoa Variabilis',
    'SID':'Siderastrea spp.',
    'HYP':'Hypnea Musciformis',
    'OMCO2':'Leathery Macrophytes: Other',
    'A/J':'Amphiroa spp.',
    'PLG':'Plexaurella Grandiflora',
    'PAC':'Palythoa Caribaeorum',
    'OSPO5':'Sponge',
    'VALV':'Valonia Ventricosa',
    'ANM':'Anemone',
    'DICTM':'Dictyota Menstrualis',
    'TUB':'Tubastrea',
    'C':'Coral',
    'GLD':'Gelidium spp.',
    'FISH':'Fish',
    'D_coral':'Dead Coral',
    'INC3':'Encrusting sponge',
    'DICPJM':'Dictyopteris Jamaicensis',
    'CAU':'Caulerpa',
    'AGH':'Agaricia Humilis',
    'OMCA':'Macroalgae: Articulated calcareous',
    'PLACO':' Placospongia'
}
