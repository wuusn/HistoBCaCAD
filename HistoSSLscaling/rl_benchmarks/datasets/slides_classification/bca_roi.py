# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""Camelyon16 dataset loading."""

from pathlib import Path
from typing import Union

import pandas as pd
import random

from ...constants import CAMELYON16_PATHS
from ...utils import merge_multiple_dataframes


def load_suqh_roi_model(
    features_root_dir: Union[str, Path],
    tile_size: int = 336,
    cohort: str = "SUQH",
    load_slide: bool = False,
) -> pd.DataFrame:
    """Load data from Camelyon16 dataset [1]_.

    Parameters
    ----------
    features_root_dir: Union[str, Path]
        Path to the histology features' root directory e.g.
        /home/user/data/rl_benchmarks_data/preprocessed/
        slides_classification/features/iBOTViTBasePANCAN/CAMELYON16_FULL/. If no
        features have been extracted yet, `features_path` is made of NaNs.
    cohort: str
        The subset of Camelyon16 cohort to use, either ``'TRAIN'`` or ``'TEST'``.
    tile_size: int = 224
        Indicate which coordinates to look for (224, 256 or 4096).
    load_slide: bool = False
        Add slides paths if those are needed. This parameter should be set
        to ``False`` if slides paths are not needed, i.e. for downstream tasks
        as only features matter, or ``True`` for features extraction (features
        have not been generated from slides yet).

    Returns
    -------
    dataset: pd.DataFrame
        This dataset contains the following columns:
        "patient_id": patient ID (is slide ID for Camelyon16)
        "slide_id": slide ID
        "slide_path": path to the slide
        "coords_path": path to the coordinates
        "label": values of the outcome to predict
    
    References
    ----------
    .. [1] https://camelyon17.grand-challenge.org/Data/ (CC0 1.0 License).
    """
    data = {}
    data["patient_id"] = [i for i in range(100)]
    data["slide_id"] = [i for i in range(100)]
    data["slide_path"] = [f"{features_root_dir}/{i}.svs" for i in range(100)]
    data["coords_path"] = None#[f"{features_root_dir}/1001-1_coords.csv"]
    data["label"] = [random.randint(0,2) for _ in range(100)]
    data['features_path'] = ['/mnt/hd0/project/bcacad/tmp/a.npy']*100
    data['center_id'] = None
    dataset = pd.DataFrame(data)
    return dataset
