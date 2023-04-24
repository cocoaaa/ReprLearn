import os
from pathlib import Path
from typing import Tuple, List, Union, Optional, Callable
from functools import partial

import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

CACHE_DIR = str(Path(os.environ.get('myssd', '~'))/'data/torch_cache')
                 
def get_resnet50_head(pretrained: bool=True,
                      cache_dir: Optional[Union[Path,str]]=CACHE_DIR):
    os.environ['TORCH_HOME'] = cache_dir
    resnet = models.resnet50(pretrained=pretrained)
    modules = list(resnet.children())[:-1]
    resnet_head = nn.Sequential(*modules)
    for p in resnet_head.parameters():
        p.requires_grad = False
    return resnet_head


def get_barlowtwin_head(pretrained:bool=True,
                      cache_dir: Optional[Union[Path,str]]=CACHE_DIR):

    pretrained_bt = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
    # get the head of resnet50 trained in barlow-twin paper
    modules = list(pretrained_bt.children())[:-1]
    bt_head = nn.Sequential(*modules)  # same as in resent50, outputs 2048-long vector

    for p in bt_head.parameters():
        p.requires_grad = False
        
    return bt_head

def extract_feature(
    extractor: nn.Module,
    input: torch.Tensor,
    device: torch.device='cpu') -> torch.Tensor:

    if input.ndim == 3:
       input = input[None, :, :, :] # (bs=1, c, h, w)
    extractor.eval().to(device)
    
    with torch.no_grad():
        z = extractor(input)
        # print(z.shape) #(bs=1, dim_z=2048, 1,1)
        z.squeeze_()
        # print(z.shape) #(dim_z=2048)

    return z.detach().to(device)
     

def extract_features(
    extractor: nn.Module,
    dataloader: DataLoader,
    batch_size: Optional[int]=36,
    max_num_samples: Optional[int]=None,
    device: Optional[torch.device]='cpu',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """returns features, labels
    
    extractor: 
        e.g., resent50_head, barlowtwin_resnet_head
    """
    extractor.eval().to(device)
    
    n_samples = max_num_samples or len(dataloader.dataset)
    print('Nsamples: ', n_samples)
    
    n_iters = int(np.ceil(n_samples/batch_size))

    features = [] 
    labels = [] 
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(dataloader):
            if i >= n_iters:
                break

            batch_z = extractor(batch_x)
    #         print(batch_z.shape) #(bs, dim_z=2048, 1,1)
            batch_z.squeeze_()
    #         print(batch_z.shape) #(bs, dim_z=2048)
            features.append(batch_z.detach().cpu())


            print('batch_y shape: ', batch_y.shape) #(bs,)
            labels.append(batch_y.detach().cpu())

    features = torch.vstack(features) # (n_samples, dim_z=2048)
    labels = torch.concat(labels)     # (n_samples,)

    return features, labels

extract_feature_resnet50 = partial(extract_feature, extractor=get_resnet50_head())
extract_feature_barlowtwin = partial(extract_feature, extractor=get_barlowtwin_head())
    
extract_features_resnet50 = partial(extract_features, extractor=get_resnet50_head())
extract_features_barlowtwin = partial(extract_features, extractor=get_barlowtwin_head())



    
# transforms for resent
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

xform_for_resnet = transforms.Compose([
#     transforms.Resize(256), #all our images in gm256 is already 256x256
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])


    
# test extract_features
def test_extract_features_resnet50(dl_for_resnet):
    resnet_head = get_resnet50_head()
    temp_feats, temp_labels = extract_features(
        extractor=resnet_head,
        dataloader=dl_for_resnet,
        max_num_samples=36
    )
    print('feats shape: ',temp_feats.shape)
    print('labels shape: ',temp_labels.shape)
    
# test extract_features
def test_extract_features_barlowtwin(dl_for_resnet):
    bt_head = get_barlowtwin_head()
    
    temp_feats, temp_labels = extract_features(
        extractor=bt_head,
        dataloader=dl_for_resnet,
        max_num_samples=36
    )
    print('feats shape: ',temp_feats.shape)
    print('labels shape: ',temp_labels.shape)
    



