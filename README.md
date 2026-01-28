hipocount-napari
================

## Quantitative analysis of immunofluorescence images of hippocampal slices

**hipocount-napari** is a plugin designed for the automated quantification of immunofluorescence signals in hippocampal brain slices. It provides a suite of widgets for image preprocessing, segmentation of morphological structures (pyramidal layer, astrocytes, GLT-1 clusters), and quantification of signal properties (area, intensity, density).

The plugin workflow typically involves:
1.  **Preprocessing**: Projection and processing of z-stack images.
2.  **Segmentation**: Generating masks for the pyramidal layer, astrocytes, and signal dots (e.g., GLT-1 clusters).
3.  **Quantification**: Counting and measuring signals within specific regions of interest.

---

## Widgets Description
### 1. Preprocessing
Preprocesses raw dual-channel z-stack images by performing projections and background subtraction.

**Functionality:**
- Separates a 4D input image into reference and target channels.
- Generates a projection for the reference channel (Maximum Intensity Projection or Average).
- Optionally applies a median filter to the reference channel.
- Subtracts background (1st percentile) from both channels.
- Converts images to `uint16`.

**Parameters:**
- **img**: Input z-stack image (4D: Channel, Z, Y, X).
- **reference_ch**: Selects which channel is the reference (`Ch.0` or `Ch.1`). The other channel is treated as the target.
- **ref_ch_processing**: Method for reference channel projection (`MIP` or `average`).
- **kernel_size**: Kernel size for the median filter applied to the reference channel. Set to `0` to disable.
- **background_substraction**: If checked, subtracts background (1st percentile) from both channels.

### 2. Pyramid layer masking
Generates a binary mask covering the neuronal pyramidal layer based on the reference signal (e.g., neuronal marker).

**Functionality:**
- Thresholds the image using Otsu's method.
- Dilates the thresholded soma regions.
- Selects the largest connected component (the layer).
- Performs a final dilation to ensure full coverage.

**Parameters:**
- **pyramid_img**: Input reference image (usually the projected result from preprocessing).
- **soma_extention**: Radius for the initial dilation of soma masks.
- **mask_extention**: Radius for the final dilation of the pyramid layer mask.

### 3. Astrocytes masking
Segments individual astrocytes from the input image.

**Functionality:**
- Combines global and local Otsu thresholding to identify astrocyte processes and somata.
- Filters out small debris.
- Optionally applies dilation to the resulting mask.
- Filters regions by minimum area.

**Parameters:**
- **astrocyte_img**: Input image containing astrocyte staining.
- **otsu_footprint_size**: Radius for the local Otsu thresholding (adaptive thresholding).
- **mask_dilation**: Radius for the dilation of the final astrocyte mask.
- **min_area_10x**: Minimum area threshold for astrocyte regions. (the value is multiplied by 10 internally).

### 4. GLT dots masking
Segments punctate signals (dots), such as GLT-1 clusters.

**Functionality:**
- Applies local Otsu thresholding to identify high-intensity puncta.
- Can optionally restrict the segmentation to a meaningful region defined by another mask.

**Parameters:**
- **dots_img**: Input image containing the dots/puncta.
- **filter_mask**: A Label layer to restrict the analysis (used if `filter_by_mask` is checked).
- **otsu_footprint_size**: Radius for the local Otsu thresholding.
- **filter_by_mask**: If checked, only dots falling within the `filter_mask` are kept.

### 5. GLT count in pyramid layer
Quantifies dots (GLT-1) specifically within the pyramidal layer.

**Functionality:**
- Calculates the total intensity and area of dots within the pyramid mask.
- Computes relative intensity and area ratios.
- Outputs results to the console and optionally saves them to a CSV file.

**Parameters:**
- **glt_img**: Input image with dot signals.
- **glt_mask**: The segmented dots mask (Labels layer).
- **pyramid_mask**: The segmented pyramid layer mask (Labels layer).
- **save_data_frame**: If checked, saves the results table to a CSV file.
- **saving_path**: Directory path where the CSV file will be saved.

**Output Data Structure (CSV)**:
- **sample** (String): Name of the analyzed image.
- **dots_intensity** (Integer): Total integrated intensity of the signal within detected dots.
- **layer_intensity** (Integer): Total integrated intensity of the signal within the entire pyramidal layer mask.
- **relative_intensity** (Float): Ratio of `dots_intensity` to `layer_intensity`.
- **dots_area** (Integer): Total area of detected dots (in pixels).
- **layer_area** (Integer): Area of the pyramidal layer mask (in pixels).
- **relative_area** (Float): Ratio of `dots_area` to `layer_area`.
- **dots_number** (Integer): Number of individual dots detected.
- **glt_mask** (String): Metadata indicating the measurement mode ('dots' or 'area').

### 6. GLT count in astrocytes
Quantifies dots (GLT-1) within individual segmented astrocytes.

**Functionality:**
- Iterates through each labeled astrocyte region.
- Calculates statistics for dots within that astrocyte (number, area, intensity, density).
- Generates a detailed report where each row represents one astrocyte.
- Saves the results to a CSV file.

**Parameters:**
- **glt_img**: Input image with dot signals.
- **glt_mask**: The segmented dots mask.
- **astrocyte_mask**: The segmented astrocytes mask (where each cell has a unique label).
- **group**: A string label to identify the experimental group in the output table.
- **saving_path**: Directory path where the CSV file will be saved.

**Output Data Structure (CSV)**:
- **id** (String): Name of the analyzed image.
- **group** (String): Experimental group label provided by the user.
- **cell_num** (Integer): Unique label ID of the astrocyte.
- **cell_area** (Integer): Area of the astrocyte (in pixels).
- **dot_num** (Integer): Number of dots detected within the astrocyte.
- **dot_area** (Integer): Total area of dots within the astrocyte (in pixels).
- **dot_rel_area** (Float): Ratio of `dot_area` to `cell_area`.
- **dot_sum_int** (Float): Total integrated intensity of all dots within the astrocyte.
- **dot_mean_int** (Integer): Mean pixel intensity of the dots.
- **dot_men_int_per_dot** (Integer): Average integrated intensity per single dot (`dot_sum_int` / `dot_num`).
- **dot_mean_int_dens** (Integer): Mean of the mean intensities of individual dots.

---

## Installation

It is recommended to use a clean `conda` environment for the installation.

1.  Create and activate a new virtual environment:

    ```bash
    conda create -n hipocount-napari-env python=3.9
    conda activate hipocount-napari-env
    ```

2.  Install `napari`:

    ```bash
    pip install "napari[all]"
    ```

3.  Install the plugin:

    ```bash
    pip install hipocount-napari
    ```

    Or, if you are installing from the source code:

    ```bash
    pip install -e .
    ```

---

## License

Distributed under the MIT License. See `LICENSE` for more information.

Copyright (c) 2024-2026 Borys Olifirov and Yana Naumenko.