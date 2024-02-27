from magicgui import magic_factory

import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker

import warnings
import pathlib
import os
import pandas as pd

import numpy as np
from scipy import ndimage as ndi
from scipy import stats

from skimage import filters
from skimage.filters import rank
from skimage import morphology
from skimage import measure
from skimage import exposure
from skimage import util

def _save_img(viewer: Viewer, img:np.ndarray, img_name:str):
    try: 
        viewer.layers[img_name].data = img
        viewer.layers[img_name].colormap = 'turbo'
    except KeyError:
        viewer.add_image(img, name=img_name, colormap='turbo')


def _get_labels_values(labels_data):
    labels_props = measure.regionprops(label_image=labels_data)
    label_values = []
    for prop in labels_props:
        label_values.append(prop['label'])
    return label_values


def _get_labels_color(labels_data):
    colors_values = _get_labels_values(labels_data)
    labels_layer = Labels(labels_data)
    colors = labels_layer.get_color(colors_values)
    return colors


@magic_factory(call_button='Preprocess z-stack',
               reference_ch={"choices": ['Ch.0', 'Ch.1']},  # reference_ch_processing={"choices": ['MIP', 'average']},
               target_ch_processing={"choices": ['MIP', 'average']},)
def stack_process(viewer: Viewer, img:Image,
                  reference_ch:str='Ch.1',  # reference_ch_processing:str='average',
                  reference_ch_gaussian_blur:bool=True, gaussian_sigma=0.75,
                  target_ch_processing:str='MIP'):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 4:
           
            if reference_ch == 'Ch.0':
                ref_index, tar_index = 0, 1
            elif reference_ch == 'Ch.1':
                ref_index, tar_index = 1, 0

            # if reference_ch_processing == 'MIP':
            #     ref_img = np.max(img.data[ref_index], axis=0)
            # elif reference_ch_processing == 'average':
            #     ref_img = np.mean(img.data[ref_index], axis=0)

            ref_img_raw = np.copy(img.data[ref_index])
            ref_img = np.mean(ref_img_raw, axis=0)
            # if reference_ch_enhance:
            #     p2, p98 = np.percentile(ref_img, (2, 98))
            #     ref_img = exposure.rescale_intensity(ref_img, in_range=(p2, p98))
            if reference_ch_gaussian_blur:
                ref_img = filters.gaussian(ref_img, sigma=gaussian_sigma)

            tar_img_raw = np.copy(img.data[tar_index])
            if target_ch_processing == 'MIP':
                tar_img = np.max(tar_img_raw, axis=0)
            elif target_ch_processing == 'average':
                tar_img = np.mean(tar_img_raw, axis=0)

            # ref_img = exposure.rescale_intensity(ref_img, in_range='dtype')
            # tar_img = exposure.rescale_intensity(tar_img, in_range=(0, 1))

            _save_img(viewer=viewer, img=tar_img, img_name=img.name + '_target')
            _save_img(viewer=viewer, img=ref_img, img_name=img.name + '_ref')
        else:
            raise ValueError('The input image should have 4 dimensions!')
        
@magic_factory(call_button='Mask somas')
def pyramid_masking(viewer:Viewer, pyramid_img:Image,
                    soma_extention:int=5,
                    mask_extention:int=5):
    def select_large_mask(raw_mask):
        element_label = measure.label(raw_mask)
        element_area = {element.area : element.label for element in measure.regionprops(element_label)}
        larger_mask = element_label == element_area[max(element_area.keys())]
        return larger_mask
    
    ref_img = pyramid_img.data

    neurons_th = filters.threshold_otsu(ref_img)
    mask_somas_raw = ref_img > neurons_th
    mask_somas = morphology.binary_dilation(mask_somas_raw, footprint=morphology.disk(soma_extention))
    mask_pyramid = select_large_mask(mask_somas)
    mask_pyramid = morphology.binary_dilation(mask_pyramid, footprint=morphology.disk(mask_extention))

    mask_name = img.name + '_pyramid-mask'
    warnings.filterwarnings('ignore')
    try:
        viewer.layers[mask_name].data = mask_pyramid.astype(bool)
    except KeyError or ValueError:
        viewer.add_labels(mask_pyramid.astype(bool), name=mask_name,
                          num_colors=1, color={1:(255,0,0,255)},
                          opacity=0.5)
        

@magic_factory(call_button='Mask astrocytes')
def astrocytes_masking(viewer:Viewer, astrocyte_img:Image,
                       otsu_footprint_size:float=100,
                       filter_by_size:bool=False,
                       min_size:float=50.0,
                       cells_extention:int=0):
    mask_name = astrocyte_img.name + '_astrocytes-mask'
    def update_astro_mask(mask):
        warnings.filterwarnings('ignore')
        try:
            viewer.layers[mask_name].data = mask
        except KeyError or ValueError:
            viewer.add_labels(mask, name=mask_name, opacity=0.5)
            
    @thread_worker(connect={'yielded': update_astro_mask})
    def _astrocytes_masking():
        ref_img = exposure.rescale_intensity(astrocyte_img.data.astype(float))
        # ref_img_adj = exposure.equalize_adapthist(ref_img, clip_limit=clip_lim)
        # mask_raw = ref_img > filters.threshold_otsu(ref_img_adj)

        ref_th = rank.otsu(ref_img, footprint=morphology.disk(otsu_footprint_size))
        mask_raw = ref_img > ref_th

        label_raw = measure.label(mask_raw)
        if filter_by_size:
            label_area = {element.label : element.area for element in measure.regionprops(label_raw)}
            filtered_mask = np.zeros_like(label_raw)
            for label_key in label_area.keys():
                if label_area[label_key] >= int(min_size * 1000):
                    filtered_mask[label_raw == label_key] = 1
            if cells_extention > 0:
                filtered_mask = morphology.dilation(filtered_mask, footprint=morphology.disk(cells_extention))
            label_astro = measure.label(filtered_mask)
        else:
            label_astro = label_raw

        print(f'Detected {np.max(label_astro)} astrocytes')
        yield label_astro

    _astrocytes_masking()


@magic_factory(call_button='Mask dots')
def otsu_dots_masking(viewer:Viewer, dots_img:Image, filter_mask:Labels,
                      otsu_footprint_size:int=50,
                      filter_by_mask:bool=True):
    mask_name = dots_img.name + '_dots'

    def update_dots_mask(mask):
        warnings.filterwarnings('ignore')
        try:
            viewer.layers[mask_name].data = mask
        except KeyError or ValueError:
            viewer.add_labels(mask, name=mask_name,
                            num_colors=1, color={1:(0,0,255,255)},
                            opacity=0.5)

    @thread_worker(connect={'yielded': update_dots_mask})
    def _otsu_dots_masking():
        glt_img = dots_img.data
        glt_th = rank.otsu(glt_img, footprint=morphology.disk(otsu_footprint_size))
        glt_mask = glt_img > glt_th
        glt_mask = morphology.opening(glt_mask, footprint=morphology.disk(1))

        if filter_by_mask:
            glt_mask[~filter_mask.data.astype(np.bool_)] = False
            yield glt_mask
        else:
            yield glt_mask

    _otsu_dots_masking()


@magic_factory(call_button='Count GLT in pyramid layer')
def pyramid_glt_count(glt_img:Image, glt_mask:Labels, pyramid_mask:Labels,
              save_data_frame:bool=False, saving_path:pathlib.Path = os.getcwd()):
    
    img = glt_img.data
    mask = glt_mask.data.astype(np.bool_)
    p_mask = pyramid_mask.data.astype(np.bool_)

    results_dict = {}
    results_dict.update({'sample':glt_img.name})

    intensity = int(np.sum(img, where=mask))
    p_intensity = int(np.sum(img, where=p_mask))
    relative_intensity = intensity / p_intensity
    results_dict.update({'dots_intensity':intensity})
    results_dict.update({'layer_intensity':p_intensity})
    results_dict.update({'relative_intensity':relative_intensity})

    glt_area = int(np.sum(mask))
    pyramid_area = int(np.sum(p_mask))
    relative_area = glt_area/pyramid_area
    results_dict.update({'dots_area':glt_area})
    results_dict.update({'layer_area':pyramid_area})
    if glt_area == pyramid_area:
        results_dict.update({'relative_area':'NA'})
    else:
        results_dict.update({'relative_area':relative_area})

    label,label_n = ndi.label(mask)
    if label_n == 1:
        results_dict.update({'dots_number':'NA'})
        results_dict.update({'glt_mask':'area'})
    else:
        
        results_dict.update({'dots_number':int(label_n)})
        results_dict.update({'glt_mask':'dots'})

    results_df = pd.DataFrame(results_dict, index=[0])

    pd.set_option('display.max_columns', None)
    print(results_df)

    if save_data_frame:
        df_name = glt_img.name
        results_df.to_csv(os.path.join(saving_path, df_name+'.csv'))


if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    viewer = Viewer()