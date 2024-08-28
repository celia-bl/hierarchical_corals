# objectives : load efficient net with the weights of coral net
# patches to feed efficient net
# get a feature vector for each patch

import torch
import torchvision.models as models
from torch import nn


def load_model(model_path):
    '''
    Load model from the path to the weights.pt
    :param model_path: the path to the weights
    :return:
    '''
    # load weights of the model pretrained by coral net team
    w_file = torch.load(model_path, map_location='cuda')

    # load efficient net
    efficientnet = models.efficientnet_b0(weights=w_file)

    # check if CUDA available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    efficientnet.to(device)

    efficientnet.eval()
    return efficientnet


def load_model_pretrained():
    efficientnet = models.efficientnet_b0(pretrained=True).cuda()

    efficientnet.eval()

    return efficientnet


