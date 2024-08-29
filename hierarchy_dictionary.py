import os
import torch
from utils import CaseInsensitiveDict
hierarchy = {
    'FISH': None,
    'Rock': None,
    'Unk': None,
    'SHAD': None,
    #cyanobacterias
    'Cyanobacteria': None,
    'CIAN4' : 'Cyanobacteria',
    'CIAN2' : 'Cyanobacteria',
    #substrates
    'Substrates': None,
    'Sand':'Substrates',
    'ARC': 'Substrates',
    #other invertebrates
    'Other invertebrates': None,
    'Sponges':'Other invertebrates',
    'OSPO5':'Sponges',
    'PLACO':'Sponges',
    'INC3':'Sponges',
    'OASC':'Other invertebrates',
    'ANM':'Other invertebrates',
    #algaes
    'Algae': None,
    # non calcified
    'Non Calcified': 'Algae',
    'PTR':'Non Calcified',
    'COD': 'Non Calcified',
    'WRAN': 'Non Calcified',
    'CLAD': 'Non Calcified',
    'OMFL1': 'Non Calcified',
    'HYP': 'Non Calcified',
    'OMCO2': 'Non Calcified',
    'SARG': 'Non Calcified',
    'VALV': 'Non Calcified',
    'LAU': 'Non Calcified',
    'Gelidiaceae': 'Non Calcified',
    'GLD': 'Gelidiaceae',
    'Gespp.': 'Gelidiaceae',
    'Caulerpa': 'Non Calcified',
    'CAU': 'Caulerpa',
    'CAU2': 'Caulerpa',
    'CAUC':'Caulerpa',
    'Dictyopteris':'Non Calcified',
    'DICMER': 'Dictyopteris',
    'DICP': 'Dictyopteris',
    'DICPP': 'Dictyopteris',
    'DICPD': 'Dictyopteris',
    'DICPJM': 'Dictyopteris',
    'Dictyota': 'Non Calcified',
    'DICTC':'Dictyota',
    'DICTM':'Dictyota',
    'CCV': 'Dictyota',
    'DICTP' : 'Dictyota',
    'DICT4': 'Dictyota',
    'Turf': 'Non Calcified',
    'TFL':'Turf',
    'Turf_sand' : 'Turf',
    'TCC' : 'Turf',
    #calcified
    'Calcified': 'Algae',
    'CNA/CC' : 'Calcified',
    'OMCA' : 'Calcified',
    'A/J' : 'Calcified',
    #corals
    'Corals': None,
    'C':'Corals',
    'Soft':'Corals',
    'Zoanthidae' : 'Soft',
    'Palythoa' : 'Zoanthidae',
    'PAC': 'Palythoa',
    'PAL' : 'Palythoa',
    'PRPV': 'Palythoa',
    'Zoanthus' : 'Zoanthidae',
    'ZO' : 'Zoanthus',
    'ZOS' : 'Zoanthus',
    'Octocoral': 'Soft',
    'Plexauridae':'Octocoral',
    'PLE':'Plexauridae',
    'PLG': 'Plexauridae',

    'Hydro-corals':'Corals',
    'Millepora': 'Hydro-corals',
    'MIL': 'Millepora',
    'MIA': 'Millepora',

    'Hard' : 'Corals',
    'Favia' : 'Hard',
    'FAV' : 'Favia',
    'FAL' : 'Favia',
    'Porites' : 'Hard',
    'POA' : 'Porites',
    'Siderastrea' : 'Hard',
    'SID' : 'Siderastrea',
    'SIDS' : 'Siderastrea',
    'Tubastrea' : 'Hard',
    'TUB' : 'Tubastrea',
    'Mussimila' : 'Hard',
    'MSP': 'Mussimila',
    'Agaricia': 'Hard',
    'AGF' : 'Agaricia',
    'AGH' : 'Agaricia',

    'Bleached' : 'Corals',
    'B_HC': 'Bleached',
    'SINV_BLC':'Bleached',
    'BL':'Bleached',
    'D_coral': 'Bleached',
    'RD_HC': 'Bleached'
}

hierarchy_MLC = {
    'Corals':None,
    'Soft':'Corals',
    'Hard':'Corals',
    'Sand':None,
    'Algae': None,
    'Macro': 'Algae',
    'Turf':'Algae',
    'CCA':'Algae',
    'Acrop':'Hard',
    'Acan':'Hard',
    'Astreo':'Hard',
    'Cypha':'Hard',
    'Favia':'Hard',
    'Fung':'Hard',
    'Gardin':'Hard',
    'Herpo':'Hard',
    'Lepta':'Hard',
    'Lepto':'Hard',
    'Lobo':'Hard',
    'Monta':'Hard',
    'Monti':'Hard',
    'Pachy':'Hard',
    'Pavon':'Hard',
    'Pocill':'Hard',
    'Porite':'Hard',
    'Porit':'Porite',
    'P. Rus':'Porite',
    'P. Irr': 'Porite',
    'P mass':'Porite',
    'Psam':'Hard',
    'Sando':'Hard',
    'Stylo':'Hard',
    'Tuba':'Hard',
    'Mille':'Soft',
    'Off':None
}

# Function to find the path to a node
def find_hierarchy_path(node):
    path = []
    while node is not None:
        path.append(node)
        node = hierarchy.get(node)
    path.reverse()  # Reverse to get path from root to the node
    return path


def duplicate_features():
    source_dir = '.features/efficientnet-b0_features'
    destination_dir = '.features/efficientnet-b0_hier_features'
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)
    if not os.path.exists(source_dir):
        raise Exception('No features encountered')
    for filename in os.listdir(source_dir):
        if filename.endswith('.pt'):
            source_path = os.path.join(source_dir, filename)
            destination_path = os.path.join(destination_dir, filename)

            data = torch.load(source_path)

            if 'labels' in data:
                data['labels'] = [find_hierarchy_path(label) for label in data['labels']]

            torch.save(data, destination_path)

