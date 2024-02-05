from magicgui import magic_factory

import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info

import pathlib
import os

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from skimage import filters
from skimage.filters import rank
from skimage import morphology
from skimage import measure

# import vispy.color

# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_qt5agg import FigureCanvas


def _save_img(viewer: Viewer, img:np.ndarray, img_name:str):
    try: 
        viewer.layers[img_name].data = img
        viewer.layers[img_name].colormap = 'turbo'
    except KeyError:
        viewer.add_image(img, name=img_name, colormap='turbo')


@magic_factory(call_button='Preprocess z-stack',
               reference_channel={"choices": ['Ch.0', 'Ch.1']},
               reference_processing={"choices": ['MIP', 'average']},
               target_processing={"choices": ['MIP', 'average']},)
def stack_process(viewer: Viewer, img:Image,
                  gaussian_blur:bool=False, gaussian_sigma=0.25,
                  reference_channel:str='Ch.1',
                  reference_processing:str='average',
                  target_processing:str='MIP'):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 4:
           
            if reference_channel == 'Ch.0':
                ref_index, tar_index = 0, 1
            elif reference_channel == 'Ch.1':
                ref_index, tar_index = 1, 0

            if reference_processing == 'MIP':
                ref_img = np.max(img.data[ref_index], axis=0)
            elif reference_processing == 'average':
                ref_img = np.mean(img.data[ref_index], axis=0)

            if target_processing == 'MIP':
                tar_img = np.max(img.data[tar_index], axis=0)
            elif target_processing == 'average':
                tar_img = np.mean(img.data[tar_index], axis=0)
            
            if gaussian_blur:
                ref_img = filters.gaussian(ref_img, sigma=gaussian_sigma)
                # tar_img = filters.gaussian(tar_img, sigma=gaussian_sigma)

            img_processed = np.stack([ref_img.astype(np.uint16), tar_img.astype(np.uint16)], axis=0)
            # img_name = img.name + '_projection'
            _save_img(viewer=viewer, img=img_processed, img_name=img.name)
        else:
            raise ValueError('The input image should have 4 dimensions!')
        
@magic_factory(call_button='Mask somas')
def pyramid_masking(viewer:Viewer, img:Image,
                    soma_extention:int=5,
                    mask_extention:int=5):
    def select_large_mask(raw_mask):
        element_label = measure.label(raw_mask)
        element_area = {element.area : element.label for element in measure.regionprops(element_label)}
        larger_mask = element_label == element_area[max(element_area.keys())]
        return larger_mask
    
    ref_img = img.data[0]

    neurons_th = filters.threshold_otsu(ref_img)
    mask_somas_raw = ref_img > neurons_th
    mask_somas = morphology.binary_dilation(mask_somas_raw, footprint=morphology.disk(soma_extention))
    mask_pyramid = select_large_mask(mask_somas)
    mask_pyramid = morphology.binary_dilation(mask_pyramid, footprint=morphology.disk(mask_extention))

    mask_name = img.name + '_pyramid-mask'
    try:
        viewer.layers[mask_name].data = mask_pyramid.astype(bool)
    except KeyError:
        viewer.add_labels(mask_pyramid.astype(bool), name=mask_name,
                          num_colors=1, color={1:(255,0,0,255)},
                          opacity=0.5)
        

@magic_factory(call_button='Mask dots')
def otsu_dots_masking(viewer:Viewer, img:Image, filter_mask:Labels,
                      otsu_footprint_size:int=10,
                      filter_by_mask:bool=True):
    glt_img = img.data[1]
    glt_th = rank.otsu(glt_img, footprint=morphology.disk(otsu_footprint_size))
    glt_mask = glt_img > glt_th
    glt_mask = morphology.opening(glt_mask, footprint=morphology.disk(1))

    if filter_by_mask:
        glt_mask[~filter_mask.data.astype(np.bool_)] = False

    mask_name = img.name + '_dots'
    try:
        viewer.layers[mask_name].data = glt_mask
    except KeyError:
        viewer.add_labels(glt_mask, name=mask_name,
                          num_colors=1, color={1:(0,0,255,255)},
                          opacity=1)


@magic_factory(call_button='Count GLT')
def glt_count(glt_img:Image, glt_mask:Labels,
              save_data_frame:bool=True, saving_path:pathlib.Path = os.getcwd()):
    
    img = glt_img.data[1]
    mask = glt_mask.data.astype(np.bool_)
    results_dict = {}
    results_dict.update({'sample':glt_img.name})

    intensity = np.sum(img, where=mask)
    results_dict.update({'intensity':intensity})

    label,label_n = ndi.label(mask)
    if label_n == 1:
        results_dict.update({'area':'NA'})
        results_dict.update({'number':'NA'})
        results_dict.update({'glt_mask':'area'})
    else:
        results_dict.update({'area':np.sum(mask)})
        results_dict.update({'number':label_n})
        results_dict.update({'glt_mask':'dots'})

    if save_data_frame:
        import pandas as pd
        df_name = glt_img.name
        results_df = pd.DataFrame(results_dict, index=[0])
        print(results_df)
        results_df.to_csv(os.path.join(saving_path, df_name+'.csv'))
    else:
        print(results_dict)


if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    viewer = Viewer()