
#%%
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib.pyplot as plt
import slideio
import os

#%%
def get_top_connected_components(tumor_mask_path, top_n):
    # Read tumor mask
    tumor_mask = Image.open(tumor_mask_path)
    
    # Label connected components
    labeled_mask, num_features = ndimage.label(tumor_mask)
    
    # Calculate sizes of each component
    component_sizes = np.bincount(labeled_mask.flatten())[1:]  # Exclude background (label 0)
    sorted_indices = np.argsort(component_sizes)[::-1]  # Sort by size, descending
    
    # Get the top n largest components
    largest_components = sorted_indices[:top_n] + 1  # Component labels are 1-indexed
    
    # Create masks for the top n components
    top_masks = [(labeled_mask == i) for i in largest_components]
    
    return top_masks

#%% read roi from WSI
def read_roi(slide_path, x, y, w, h, target_mag):
    slide = slideio.open_slide(slide_path, 'SVS')
    scene = slide.get_scene(0)
    scene_mag = scene.magnification
    mag_scale = scene_mag / target_mag
    roi = scene.read_block((int(x*mag_scale), int(y*mag_scale), int(w*mag_scale), int(h*mag_scale)), (w, h))
    return roi

#%%
def get_mag(slide_path):
    slide = slideio.open_slide(slide_path, 'SVS')
    scene = slide.get_scene(0)
    return scene.magnification

#%% get ROIs from Top N connected components
def get_top_n_rois(slide_path, tumor_mask_path, top_n, target_mag, min_size, max_size, save_dir=None, mask_scale=4):
    # Get top N connected components
    top_masks = get_top_connected_components(tumor_mask_path, top_n)
    name = os.path.basename(slide_path).replace('.svs','')
    print('Processing', name)
    if save_dir:
        save_dir = f'{save_dir}/{name}'
        os.makedirs(save_dir, exist_ok=True)
    # Get ROIs for each component
    rois = []
    for i, mask in enumerate(top_masks):
        # Get centroid of mask
        y, x = ndimage.center_of_mass(mask)
        x, y = int(x), int(y)

        # get bounding box
        x_min, x_max = np.min(np.where(mask)[1]), np.max(np.where(mask)[1])
        y_min, y_max = np.min(np.where(mask)[0]), np.max(np.where(mask)[0])

        w, h = x_max - x_min, y_max - y_min
        w = min(max_size, w)
        h = min(max_size, h)
        w = max(min_size, w)
        h = max(min_size, h)
        x_min = max(0, x - w // 2)
        y_min = max(0, y - h // 2)
        
        # Get ROI from WSI
        x_min = x_min * mask_scale
        y_min = y_min * mask_scale
        w = w * mask_scale
        h = h * mask_scale
        roi = read_roi(slide_path, x_min,y_min,w,h, target_mag)
        rois.append(roi)
        if save_dir:
            save_name = f'{save_dir}/{name}_10x_{i}_x{x_min}_y{y_min}_w{w}_h{h}.png'
            Image.fromarray(roi).save(save_name)
    print('Done', name)
    return rois

# %%
def save_rois(svs_name, save_dir, rois):
    save_dir = f'{save_dir}/{svs_name}'
    os.makedirs(save_dir, exist_ok=True)
    for i, roi in enumerate(rois):
        roi_path = f'{save_dir}/{svs_name}_{i}.png'
        Image.fromarray(roi).save(roi_path)

# %%
def test():
    wsi_path = '/mnt/hd1/BJSZHP/2024-11-12/2265917A2-HE.svs'
    mask_path = '/mnt/hd0/project/bcacad/tumor_mask/bjszhp/tumor_merge/2265917A2-HE.png'
    save_dir = '/mnt/hd0/project/bcacad/data/roi-level/bjszhp'
    topN = 5
    target_mag = 10
    min_size = 336
    max_size = 1024

    rois = get_top_n_rois(wsi_path, mask_path, topN, target_mag, min_size, max_size, save_dir)
    # save_rois(os.path.basename(wsi_path).replace('.svs',''), save_dir, rois)

    # %% show top n masks
    # top_masks = get_top_connected_components(mask_path, topN)
    # fig, axes = plt.subplots(1, topN, figsize=(20, 20))
    # for i, mask in enumerate(top_masks):
    #     axes[i].imshow(mask)
    #     axes[i].axis('off')

    # %%

if __name__ == '__main__':
    # multiprocessing
    import multiprocessing
    import time
    from functools import partial
    from glob import glob

    topN = 5
    target_mag = 10
    min_size = 336
    max_size = 1024

    wsi_dir = "/mnt/hd1/BJSZHP/2024-11-12/"
    mask_dir = "/mnt/hd0/project/bcacad/tumor_mask/bjszhp/tumor_merge/"
    save_dir = "/mnt/hd0/project/bcacad/data/roi-level/bjszhp"

    wsi_paths = glob(f'{wsi_dir}/*.svs')

    mask_paths = [f'{mask_dir}/{os.path.basename(wsi_path).replace(".svs",".png")}' for wsi_path in wsi_paths]

    start = time.time()
    with multiprocessing.Pool(40) as pool:
        func = partial(get_top_n_rois, top_n=topN, target_mag=target_mag, min_size=min_size, max_size=max_size, save_dir=save_dir)
        pool.starmap(func, zip(wsi_paths, mask_paths))
    end = time.time()
    print(f'Elapsed time: {end-start}')

