# Foundation Model-powered Computer-aided Diagnosis System to Assist Multi-task Histopathological Diagnosis in Breast Cancer

This repository contains the codebase used for the HistoBCaCAD pipeline, including:
- Whole-slide tissue segmentation,
- Self-supervised foundation model finetuning,
- Multi-task patch-level finetuning/testing,
- ROI feature extraction and MIL-based ROI/WSI modeling.

> **Status note**
> This README is focused on practical usage and reproducibility of the current repository layout. File names and paths below match the current codebase.

---

## 1) Repository structure (what each module is for)

- `segmentation_model/`  
  WSI segmentation code and inference entrypoint.

- `ssl_finetune/`  
  Self-supervised learning (SSL) finetuning code (iBOT-based workflow).

- `d_swin_4_multi_task/`  
  Multi-task patch-level model training/testing, including LoRA finetuning and patch-level comparison/testing, feature extraction.

- `HistoSSLscaling/`  
  MIL training/testing notebooks/scripts for ROI-level and WSI-level prediction.

- `mil_models/`  
  Local storage location for ROI and WSI MIL model artifacts/checkpoints.

- `example_rois/`  
  Example ROI data for quick experimentation.

---


## 2) Input data directories and expected structure

Reviewer feedback asked for a clearer description of required input folders. This section provides the expected directory layouts used by the current scripts.

> **Important:** Large training datasets are **not** included in this repository. You must point scripts to your local data paths.

### A. WSI segmentation input (for `segmentation_model/run_wsi_infer.sh`)

The segmentation runner expects:
- `wsi_dir`: directory containing whole-slide files (default extension used in script is `.svs`).
- `save_dir`: directory where per-scale segmentation outputs are written.
- `merge_dir`: directory for merged multi-resolution outputs.

Example layout:

```text
/path/to/wsi_dir/
  case_001.svs
  case_002.svs
  ...
```

### B. ROI image input (for quick inference demo)

`HistoSSLscaling/example_roi_inference.py` expects a directory of ROI image files (png/jpg/tif/etc.) via `--roi_dir`.
The script scans files directly under that folder.

Example layout:

```text
/path/to/roi_dir/
  roi_0001.png
  roi_0002.png
  roi_0003.tif
  ...
```

### C. ROI dataset layout used by legacy feature extraction workflow

The legacy script `d_swin_4_multi_task/lora_feature_extraction.py` contains cohort-specific assumptions:

- For cohort `bjszhp`:
  - input expected at `data_root/bjszhp/`
  - images are typically grouped by label subfolder.
- For other cohorts listed in the script:
  - input expected at `data_root/<cohort>/test/`

Example layout:

```text
/path/to/data_root/
  bjszhp/
    normal/
      xxx.png
    dcis1/
      yyy.png
  qduh/
    test/
      roi_a.png
      roi_b.png
  shsu/
    test/
      roi_c.png
```

Output feature files are saved as `.npy` under `save_root/<cohort>/test/...`.

### D. MIL model inputs

- Pretrained MIL checkpoints used by the quick ROI demo are expected under `mil_models/` (default: `mil_models/abmil_roi.pth`).
- ROI/WSI MIL notebooks in `HistoSSLscaling/` expect extracted feature directories prepared in previous steps.

---

## 3) Main entrypoints

Use the following scripts/notebooks as the primary entrypoints:

### A. Segmentation inference
- **Script:** `segmentation_model/run_wsi_infer.sh`
- **Purpose:** Run WSI inference for the segmentation model.

### B. SSL foundation model finetuning
- **Script:** `ssl_finetune/main_ibot.py`
- **Purpose:** Main training/finetuning entrypoint for SSL (iBOT) model adaptation.

### C. LoRA finetuning (multi-task patch model)
- **Script:** `d_swin_4_multi_task/train_lora_with_ft.py`
- **Purpose:** Train/fine-tune the patch multi-task model with LoRA.

### D. Patch-level testing/comparison
- **Script:** `d_swin_4_multi_task/test_compare.py`
- **Purpose:** Evaluate patch-level model performance and run comparison testing.

### E. ROI feature extraction
- **Notebook:** `d_swin_4_multi_task/lora_feature_extraction.py`
- **Purpose:** Extract ROI features used by downstream MIL models.

### F. ROI-level MIL training (legacy)
- **Notebook:** `HistoSSLscaling/mil_roi_model_on_the_fly_lora.ipynb`
- **Purpose:** Archived ROI-level MIL training/testing notebook with saved outputs.

### G. WSI-level MIL training/testing
- **Notebook:** `HistoSSLscaling/mil_wsi_model_on_the_fly_lora.ipynb`
- **Purpose:** Train and test WSI-level MIL models with saved outputs.
---

## 4) Model weights and checkpoints

- **ROI and WSI MIL models:** stored in this repository under `mil_models/`.
- **Patch model + segmentation model:** hosted on Hugging Face:  
  https://huggingface.co/yuxinwu/histobcacad/tree/main
- Some scripts may need pretrained Phikon model weight to initialize, [download here](https://drive.google.com/drive/folders/1wIrLw4KZa8oI3hZVykH1dyvXu08_WwmL?usp=drive_link).  

---

## 5) Environment setup

All the training and experiments were done in a workstation with Rocky Linux 9.4, 2 x GeForce RTX 4090, CUDA Version 12.2. 

This repository currently uses **two environment tracks**:

1. **Segmentation environment (inside `segmentation_model/`)**
2. **Non-segmentation environment (root-level, used by SSL / patch / ROI / WSI workflows)**

### 5.1 Segmentation model environment

Environment files for segmentation are in the subfolder with Python 3.8.18:
- Conda: `segmentation_model/environment_conda.yml`
- Pip: `segmentation_model/requirements.txt`

Recommended setup:

```bash
# from repo root
conda env create -f segmentation_model/environment_conda.yml
conda activate mmseg
pip install -r segmentation_model/requirements.txt
```

Then run segmentation inference via:

```bash
bash segmentation_model/run_wsi_infer.sh
```

### 5.2 Root environment for the rest of the pipeline

For SSL finetuning, LoRA/patch training/testing, ROI feature extraction, and ROI/WSI MIL notebooks, use root-level environment files, with Python 3.8.3:
- Conda: `environment_conda.yml`
- Pip: `requirements_pip.txt`

Recommended setup:

```bash
# from repo root
conda env create -f environment_conda.yml
conda activate histosslscaling
pip install -r requirements_pip.txt
```

> If your local environment name differs, keep using the environment name defined in your local conda file.

### 5.3 Original projects used in this work
To reproduce the environment, we kept most of the files from the original projects at the time we ran the experiments.
Some of the code or environments might be out of date. Please also check these original repos used in our project:  
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[mmpretrain](https://github.com/open-mmlab/mmpretrain)  
[iBOT](https://github.com/bytedance/ibot)  
[HistoSSLscaling](https://github.com/owkin/HistoSSLscaling)


---
## 6) Quick ROI Inference Example
- **Script:** `HistoSSLscaling/example_roi_inference.py`
- **Purpose:** Run a standalone quick ROI inference demo (tile feature extraction + ABMIL inference) on `example_rois/` and print predictions.
- **Example command:**

```bash
python HistoSSLscaling/example_roi_inference.py --ibot_weights /path/to/ibot_vit_base_pancan.pth
```
with our feature extraction:
```bash
python HistoSSLscaling/example_roi_inference.py --encoder_ckpt /path/to/model-13_ft_lora.pth  --ibot_weights /home/yuxin/Downloads/ibot_vit_base_pancan.pth
```

## 7) Suggested end-to-end workflow

The recommended execution order is:

1. **Run segmentation on WSIs**  
   Use `segmentation_model/run_wsi_infer.sh` to generate segmentation outputs.

2. **SSL finetuning**  
   Use `ssl_finetune/main_ibot.py` to finetune foundation representations.

3. **Train patch multi-task model with LoRA**  
   Use `d_swin_4_multi_task/train_lora_with_ft.py`.

4. **Run patch-level testing/comparison**  
   Use `d_swin_4_multi_task/test_compare.py`.

5. **Extract ROI features**  
   Run `d_swin_4_multi_task/lora_feature_extraction.py`.

6. **Train ROI MIL model**
   Run `HistoSSLscaling/legacy/mil_roi_model_on_the_fly_lora.ipynb`.

7. **Train/test WSI MIL model**  
   Run `HistoSSLscaling/mil_wsi_model_on_the_fly_lora.ipynb`.

---

## 8) Citation / publication status

The study manuscript metadata and full abstract details can be updated here after publication.
