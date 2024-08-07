import warnings
import pathlib
import os
import pandas as pd
from timeit import default_timer as timer

import napari
from napari import Viewer
from napari.layers import Image, Labels
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker

from magicgui import magic_factory

import numpy as np
from scipy import ndimage as ndi

from skimage import filters
from skimage.filters import rank
from skimage import morphology
from skimage import measure


def _save_img(viewer: Viewer, img:np.ndarray, img_name:str):
    try: 
        viewer.layers[img_name].data = img
        viewer.layers[img_name].colormap = 'turbo'
    except KeyError:
        viewer.add_image(img, name=img_name, colormap='turbo')


def _uint_convert(img):
    if isinstance(img[0,0], np.uint16):
        return img
    else:
        i_min = img.min()
        i_max = img.max()

        a = 65535 / (i_max-i_min)
        b = 65535 - a * i_max
        uint_img = (a * img + b).astype(np.uint16)
        return uint_img


@magic_factory(call_button='Preprocess z-stack',
               reference_ch={"choices": ['Ch.0', 'Ch.1']},  # reference_ch_processing={"choices": ['MIP', 'average']},
               ref_ch_processing={"choices": ['MIP', 'average']},)
def stack_process(viewer: Viewer, img:Image,
                  reference_ch:str='Ch.1',  # reference_ch_processing:str='average',
                  kernel_size:int=1,
                  background_substraction:bool=True,
                  ref_ch_processing:str='MIP'):
    if input is not None:
        series_dim = img.data.ndim
        if series_dim == 4:
           
            if reference_ch == 'Ch.0':
                ref_index, tar_index = 0, 1
            elif reference_ch == 'Ch.1':
                ref_index, tar_index = 1, 0

            ref_img_raw = np.copy(img.data[ref_index])

            if ref_ch_processing == 'MIP':
                ref_img = np.max(ref_img_raw, axis=0).astype(ref_img_raw.dtype)
            elif ref_ch_processing == 'average':
                ref_img = np.mean(ref_img_raw, axis=0).astype(ref_img_raw.dtype)

            if kernel_size != 0:
                ref_img = filters.median(ref_img, footprint=morphology.disk(kernel_size))

            tar_img = np.max(img.data[tar_index], axis=0)
            if background_substraction:
                bc_p = lambda x: np.array([f - np.percentile(f, 1) for f in x]).clip(min=0).astype(dtype=x.dtype)
                ref_img = bc_p(ref_img)
                tar_img = bc_p(tar_img)

            _save_img(viewer=viewer, img=_uint_convert(tar_img), img_name=img.name + '_target')
            _save_img(viewer=viewer, img=_uint_convert(ref_img), img_name=img.name + '_ref')
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

    mask_name = pyramid_img.name + '_pyramid-mask'
    warnings.filterwarnings('ignore')
    try:
        viewer.layers[mask_name].data = mask_pyramid.astype(bool)
    except KeyError or ValueError:
        viewer.add_labels(mask_pyramid.astype(bool), name=mask_name,
                          num_colors=1, color={1:(255,0,0,255)},
                          opacity=0.5)
        

@magic_factory(call_button='Mask astrocytes',)
def astrocytes_masking(viewer:Viewer, astrocyte_img:Image,
                       otsu_footprint_size:int=5,
                       mask_dilation:int=2,
                       min_area_10x:int=50):
    mask_name = astrocyte_img.name + '_astrocytes-mask'
    def update_astro_mask(mask):
        warnings.filterwarnings('ignore')
        try:
            viewer.layers[mask_name].data = mask
        except KeyError or ValueError:
            viewer.add_labels(mask, name=mask_name, opacity=0.5)
            
    @thread_worker(connect={'yielded': update_astro_mask})
    def _astrocytes_masking():
        tic = timer()

        ref_img = astrocyte_img.data

        high_mask = ref_img > filters.threshold_otsu(ref_img)

        gfap_th = rank.otsu(ref_img, footprint=morphology.disk(otsu_footprint_size))
        low_mask = ref_img > gfap_th

        filter_lab, filter_lab_num = ndi.label(low_mask)
        hyst_img =  ndi.sum(high_mask, filter_lab, np.arange(filter_lab_num + 1))
        connected_mask = hyst_img > 0
        debris_mask = connected_mask[filter_lab]
        debris_mask = morphology.erosion(debris_mask, footprint=morphology.disk(5))

        astro_mask = np.copy(low_mask)
        astro_mask[~debris_mask] = 0

        if mask_dilation != 0:
            astro_mask = morphology.dilation(astro_mask, footprint=morphology.disk(mask_dilation))

        if min_area_10x != 0:
            inter_mask = np.zeros_like(astro_mask)
            inter_lab, inter_lab_num = ndi.label(astro_mask)
            show_info(f'{astrocyte_img.name}: Filtering start: {inter_lab_num} labels')

            for inter_region in measure.regionprops(inter_lab):
                inter_region_mask = inter_lab == inter_region.label
                inter_region_area = np.sum(inter_region_mask, dtype=ref_img.dtype)
                # inter_region_val = np.mean(ref_img, where=inter_region_mask, dtype=ref_img.dtype)
                # inter_region_ratio = inter_region_val / inter_region_area

                if inter_region_area >= int(min_area_10x*10):
                    inter_mask[inter_region_mask] = 1
            astro_mask = inter_mask

        astro_label = measure.label(astro_mask)
        tok = timer()

        show_info(f'{astrocyte_img.name}: Detected {np.max(astro_label)} astrocytes in {round(tok - tic,1)}s')
        yield astro_label

    _astrocytes_masking()


@magic_factory(call_button='Mask dots')
def otsu_dots_masking(viewer:Viewer, dots_img:Image, filter_mask:Labels,
                      otsu_footprint_size:int=10,
                      filter_by_mask:bool=True):
    mask_name = dots_img.name + '_dots'

    def update_dots_mask(mask):
        warnings.filterwarnings('ignore')
        try:
            viewer.layers[mask_name].data = mask
        except KeyError or ValueError:
            viewer.add_labels(mask, name=mask_name,
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


@magic_factory(call_button='Count GLT in astrocytes')
def astro_glt_count(glt_img:Image, glt_mask:Labels, astrocyte_mask:Labels, group:str='Group 1',
                    saving_path:pathlib.Path = os.getcwd()):
    g_img = glt_img.data
    g_mask = glt_mask.data
    a_mask = astrocyte_mask.data

    if g_img.ndim != 2 or g_mask.ndim != 2 or a_mask.ndim != 2:
        raise ValueError('Incorrect input data shape!')
    
    output_data_frame = pd.DataFrame({'id':[],
                                      'group':[],
                                      'cell_num':[],
                                      'cell_area':[],
                                      'dot_num':[],
                                      'dot_area':[],
                                      'dot_rel_area':[],
                                      'dot_sum_int':[],
                                      'dot_mean_int':[],
                                      'dot_men_int_per_dot':[],
                                      'dot_mean_int_dens':[]})
    
    for a_region in measure.regionprops(a_mask):
        one_a_mask = a_mask == a_region.label
        
        one_g_mask = np.copy(g_mask)
        one_g_mask = one_g_mask != 0
        one_g_mask[~one_a_mask] = 0

        one_cell_area = np.sum(one_a_mask)

        one_dot_area = np.sum(one_g_mask)
        one_dot_rel_area = one_dot_area / one_cell_area

        one_dot_lab, one_dot_num = ndi.label(one_g_mask)

        one_dot_int_dens_list = []
        for dot_lab_num in range(one_dot_lab.max()):
            one_dot_mask = one_dot_lab == dot_lab_num
            one_dot_int_dens_list.append(np.sum(g_img, where=one_dot_mask) / np.sum(one_dot_mask))
        one_dot_mean_int_dens = np.mean(np.array(one_dot_int_dens_list))

        one_dot_sum_int = np.sum(g_img, where=one_g_mask)
        one_dot_mean_int = np.mean(g_img, where=one_g_mask)

        cell_row = [glt_img.name,                        # id
                    group,                               # group
                    a_region.label,                      # cell_num
                    one_cell_area,                       # cell_area
                    one_dot_num,                         # dot_num
                    one_dot_area,                        # dot_area
                    round(one_dot_rel_area, 3),          # dot_rel_area
                    one_dot_sum_int,                     # dot_sum_int
                    int(one_dot_mean_int),               # dot_mean_int
                    int(one_dot_sum_int / one_dot_num),  # dot_men_int_per_dot
                    int(one_dot_mean_int_dens)]          # dot_mean_int_dens


        output_data_frame.loc[len(output_data_frame.index)] = cell_row
    
    output_data_frame.to_csv(os.path.join(saving_path, f'{glt_img.name}_glt_in_astro.csv'))
    show_info(f'{glt_img.name}: Astrocytes data frame saved')


if __name__ == '__main__':

    import napari
    viewer = napari.Viewer()
    viewer = Viewer()