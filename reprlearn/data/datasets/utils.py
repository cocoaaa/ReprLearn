from collections import defaultdict
from typing import Tuple, List, Union, Optional, Callable
from pathlib import Path

from PIL.Image import Image
from torchvision import datasets
import numpy as np
import pandas as pd

from reprlearn.utils.misc import is_img_fp

def get_mnist_data(
        data_root: Union[Path,str],
        use_train_dataset: bool = True,
        download=True
) -> Tuple[List[Image], List[int]]:
    # Get all imgs and digits in MNIST dataset
    ds = datasets.MNIST(data_root, train=use_train_dataset, download=download)

    pil_imgs = []
    digit_labels = []
    for i in range(len(ds)):
        x, y = ds[i]
        pil_imgs.append(x)
        digit_labels.append(y)
    return pil_imgs, digit_labels


# === torchvision Dataset from data in Pandas dataframe ===
#
# Helpers: 
def create_subdf(df: pd.DataFrame, 
                 col_groupby: str, 
                 col_sortvalues: str,
                 shuffle: bool, 
                 seed: Optional[int]=123,
                 max_n_per_group: Optional[int]=None,
                 reset_index:Optional[bool]=True,
                 **kwargs_sort_values
                )->pd.DataFrame:
    """Create a sub dataframe with max_n number of rows from each group.
   
    Args:
    - col_groupby (str)      : column to groupby the df
        - eg: label/class name (e.g. 'model_name')
    - col_sortvalues (str)   : column name to sort values within each group
        - eg: "img_fp"
    - shuffle     (bool)     : shuffle each group_df before taking the subset of rows
    - seed (int)             : random seed for shuffling each group_df's rows
    - max_n_per_class (int)  : number of images to load from each class. Default is None.
        If None, use all images in each group 
        
        
    Note: 
    - we sort each group_df by the values in the columne, `col_sortvalues`, 
    and, if shuffle, shuffle the rows in each group_df
    and, then take the first max_n_per_class to create the df.
    """
    
    if kwargs_sort_values is None:
        kwargs_sort_values = {
            'ascending': True,
            'inplace': False,
            'ignore_index': True
        }

        
    subgroups = [ ] 
    for g_name, df_g in df.groupby(col_groupby):
#         print(g_name, len(df_g)) #debug
        df_g = df_g.sort_values(by=col_sortvalues,**kwargs_sort_values)
        if shuffle:
            df_g = df_g.sample(frac=1, random_state=seed).reset_index(drop=True)
           
        if max_n_per_group is not None: 
            df_g = df_g[:max_n_per_group]
        subgroups.append(df_g)
    subdf = pd.concat(subgroups)

    if reset_index:
        subdf.reset_index(inplace=True, drop=True)
    return subdf
    

                
def create_dataset_df_from_root_dir(data_root: Path,
             is_valid_dir:Optional[Callable[[str], bool]]=None)-> pd.DataFrame:
    """Walk down each subdir of data_root_dir in sorted order, and 
    collect the path to images under each subdir, also in sorted order.
    Returns a dataFrame with columns of ['fp', 'class_name']
    
    Usage: general data_root with subdirectories as class-dires, each of which 
    contains images of the class
    
    Returns:
        pd.DataFrame with columns ['fp', 'class_name']
        
    """
    if is_valid_dir is None:
        # is_valid_dir = lambda x:True
        is_valid_dir = lambda fp: not fp.startswith('.')
    # print('str(.): is is valid?: ', is_valid_dir('.')) #debug
    
    subdirs = sorted([p for p in data_root.iterdir() 
                      if p.is_dir() and is_valid_dir(str(p))])
    # print('subdirs sorted: ', subdirs) #debug
              
    all_fps = [] 
    all_classnames = []
    for subdir in subdirs:
        fps = sorted([fp for fp in subdir.iterdir() if is_img_fp(fp)])
        classnames = [subdir.name]*len(fps)
        
        all_fps.extend(fps)
        all_classnames.extend(classnames)
        
        # debug
        # print(subdir.name, classnames[0], len(fps))
        
    data = {'fp': all_fps,
            'class_name': all_classnames}
    return pd.DataFrame.from_dict(data)


def create_dataset_df_from_gm_root_dir(
    data_root: Path, 
    n_imgs_per_subdir: Optional[int]=None,
    use_model_fullname: Optional[bool]=False,
    is_valid_dir:Optional[Callable[[str], bool]]=None,
    verbose=False) -> pd.DataFrame:

    """Create a dataframe containing information to load images(x) and labels(y)
    from a dataset root folder (`data_root`).
    Assued structure of the data_root folder:
    
    `data_root`:
    |-- real
        |-- <img1>.png
        |-- <img2>.png
        |-- ...
    |-- model1          # <fam_name>-<model-name>
        |-- <img1>.png  # any image name is okay.
        |-- <img2>.png
        |-- ...
    |-- model2
        |-- <img1>.png
        |-- <img2>.png
    |-- ...
    |-- modelk
        |-- <img1>.png
        |-- <img2>.png
        
    Further, we assume each model dir is named in the format of <fam_name>-<method_name>
    
    We Walk down each subdir of data_root_dir in sorted order, and 
    collect the path to images under each subdir, also in sorted order.
    
    Args:
    - data_root (Path)             : fullpath to the root dir containing model_dir's 
    - n_imgs_per_subdir (int)      : num. of images to use in the sorted img_fps in each subdir
        If not specified (default None), use all images in each subdir
    - use_model_fullname (bool)    : if true, use `model_name` as the fullname of <fam_name>-<model_name> (ie. the subdir for that model)
    
    Returns:
        pd.DataFrame with columns ['img_fp', 'fam_name', 'model_name']  where:
            - img_fp (str): path to the image
            - fam_name (str):  name of the model family in ['vae', 'gan', 'real', 'flow', 'score])
            - model_name (str):  name of the generative method 
                (e.g., 'betavae', 'dfcvae', 'dcgan', 'glow', 'ncsn')
    """
    if is_valid_dir is None:
        # is_valid_dir = lambda x:True
        is_valid_dir = lambda fp: not (str(fp).startswith('.') and fp.is_file())
    # print('str(.): is is valid?: ', is_valid_dir('.')) #debug
    
    subdirs = sorted([p for p in data_root.iterdir() 
                      if p.is_dir() and is_valid_dir(p)])
    # print('subdirs sorted: ', subdirs) #debug
              
    all_img_fps = [] 
    all_fam_names = []
    all_model_names = []
    for subdir in subdirs:
        model_fullname = subdir.name
        fam_name, model_name = model_fullname.split('-') # assumed format of <fam_name>-<method_name>
        
        if use_model_fullname:
            model_name = model_fullname
        
        fps = sorted([str(fp) for fp in subdir.iterdir() if is_img_fp(fp)])
        
        if n_imgs_per_subdir is not None:
            fps = fps[:n_imgs_per_subdir]
            
        fam_names = [fam_name]*len(fps)
        model_names = [model_name]*len(fps)
        # todo (?)
        # model_ids
        # fam_ids
        
        all_img_fps.extend(fps)
        all_fam_names.extend(fam_names)
        all_model_names.extend(model_names)
 
        # debug
        # print(subdir.name, classnames[0], len(fps))

            
        if verbose:
            print()
            print('model_full: ', model_fullname)
            print('model_name: ', model_name)
            print('fam_name: ', fam_name)
            print('n imgs from subdir: ', len(fps))
            print('Done: ', fam_name, model_name)
        
    data = {'img_fp': all_img_fps,
            'fam_name': all_fam_names,
            'model_name': all_model_names}
    return pd.DataFrame.from_dict(data)
   
def stratified_split(
    df_all, 
    label_key: str, # one of the column names in df_all.columns
    n_train_per_label: int,
    n_test_per_label: int,        
    shuffle: Optional[bool]=False,
    seed: Optional[int]=123,
    reset_index: Optional[bool]=True,
    verbose: Optional[bool]=False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given a dataframe, split the dataframe into df_train and df_test by
    selecting n_train_per_label number of indices (per label) and 
    n_test_per_label num. of indices per label from df_all 
    
    Args:
    - df_all (pd.DataFrame) : df 
    - label_key (str): one of the string in df_all.columns that define what we consider as label
    - n_train_per_label: num. of rows in each label for df_train
    - n_test_per_label: num. of rows per label to be collected to df_test
    - shuffle (bool) : if shuffle, shuffle in the index of df_all before spliting into two.
        Otherwise, split the indices in the order of the given df_all.
    - seed (int): random seed for shuffling. Not used if shuffle is False
    - reset_index (bool): if true, reset indices of the created split df's. 
    Returns:
    -------
    (df_train, df_test) : preserving the same structure of df_all (ie. containing data in all columns)
    
    """
    # 1. create a dict of list of all indices belonging to each label
    # ie. {'label1: [1, 100,21, ...],
    #      'label2: [213, 91, 14, ...],
    #       ... 
    #      `labelK: [5,25,32, 151, ...]}
    inds_per_label = defaultdict(list)
    for idx, row in df_all.iterrows():
    #     img_fp = row['img_fp']
    #     model_name = row['model_name']
    #     fam_name = row['fam_name']
    #     print(img_fp)
    #     print(model_name, fam_name)
        inds_per_label[row[label_key]].append(idx)
        
    # 2. split the inds_per_class for train and test
    # -- first shuffle each index list per label 
    if shuffle:
        np.random.seed(seed)
        for label, inds in inds_per_label.items():
        #     print(f'\n{label}')
        #     print(inds[:5])
            np.random.shuffle(inds)
        #     print('check if shuffled: ', inds[:5], inds_per_label[label][:5])
            # great, np.shuffle (in-place) still effects linger on the dictionary's value (list of shuffled inds)


    train_inds_per_label = {
        class_name: inds[:n_train_per_label] 
        for class_name, inds in inds_per_label.items()
    }

    test_inds_per_label = {
        class_name: inds[n_train_per_label:n_train_per_label + n_test_per_label] 
        for class_name, inds in inds_per_label.items()
    }

    # make sure they are ok:
    if verbose:
        print('\nFor train inds per label: ')
        for name,inds in train_inds_per_label.items():
            print(name, len(inds), inds[:5])
            if len(inds) != n_train_per_label:
                print(f"WARNING: {name} does not have n_train_per_label rows: {len(inds)}")

        print('\nFor test inds per label: ')
        for name,inds in test_inds_per_label.items():
            print(name, len(inds), inds[:5])
            if len(inds) != n_test_per_label:
                print(f"WARNING: {name} does not have n_test_per_label rows: {len(inds)}")
    # Good if each class in training index dict contains n_train_per_class inds!
    # Good if each class in testing index dict contains n_test_per_class inds!
    
    # 3. Create df_train, df_test from each of the dict of inds-per-class
    df_train = pd.DataFrame() # initialize with empty df
    for label, inds in train_inds_per_label.items():
        # grab a subset of df_all using this indices
        df_train = pd.concat([df_train, df_all.iloc[inds]]) 
        
    df_test = pd.DataFrame() # initialize with empty df
    for label, inds in test_inds_per_label.items():
        # grab a subset of df_all using this indices
        df_test = pd.concat([df_test, df_all.iloc[inds]]) 
    
    
    if reset_index:
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)

    # -- verify
    if verbose:
        print('df_train: ', df_train.groupby(label_key).count())
        print('df_test: ', df_test.groupby(label_key).count())
    
    return df_train, df_test 
    

        
        


        
    
    