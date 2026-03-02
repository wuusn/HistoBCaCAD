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
  Multi-task patch-level model training/testing, including LoRA finetuning and patch-level comparison/testing.

- `HistoSSLscaling/`  
  ROI feature extraction and MIL training/testing notebooks/scripts for ROI-level and WSI-level prediction.

- `mil_models/`  
  Local storage location for ROI and WSI MIL model artifacts/checkpoints.

- `example_rois/`  
  Example ROI data for quick experimentation.

---

## 2) Main entrypoints

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
- **Notebook:** `HistoSSLscaling/extract_roi_features.ipynb`
- **Purpose:** Extract ROI features used by downstream MIL models.

### F. ROI-level MIL training (legacy)
- **Notebook:** `HistoSSLscaling/mil_roi_model_on_the_fly_lora.ipynb`
- **Purpose:** Archived ROI-level MIL training/testing notebook with saved outputs.

### G. WSI-level MIL training/testing
- **Notebook:** `HistoSSLscaling/mil_wsi_model_on_the_fly_lora.ipynb`
- **Purpose:** Train and test WSI-level MIL models with saved outputs.

---

## 3) Model weights and checkpoints

- **ROI and WSI MIL models:** stored in this repository under `mil_models/`.
- **Patch model + segmentation model:** hosted on Hugging Face:  
  https://huggingface.co/yuxinwu/histobcacad/tree/main

---

## 4) Environment setup

All the training and experiments were done in a workstation with Rocky Linux 9.4, 2 x GeForce RTX 4090, CUDA Version 12.2. 

This repository currently uses **two environment tracks**:

1. **Segmentation environment (inside `segmentation_model/`)**
2. **Non-segmentation environment (root-level, used by SSL / patch / ROI / WSI workflows)**

### 4.1 Segmentation model environment

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

### 4.2 Root environment for the rest of the pipeline

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

### 4.3 Original projects used in this work
Some of the code or environments might be out of date. Please also check these original repos used in our project:  
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)  
[mmpretrain](https://github.com/open-mmlab/mmpretrain)  
[iBOT](https://github.com/bytedance/ibot)  
[HistoSSLscaling](https://github.com/owkin/HistoSSLscaling)


---

## 5) Suggested end-to-end workflow

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
   Run `HistoSSLscaling/extract_roi_features.ipynb`.

6. **Train ROI MIL model**
   Run `HistoSSLscaling/legacy/mil_roi_model_on_the_fly_lora.ipynb`.

7. **Train/test WSI MIL model**  
   Run `HistoSSLscaling/mil_wsi_model_on_the_fly_lora.ipynb`.

---

## 6) Practical notes

- Legacy/backup notebooks and scripts are archived under `legacy/` subfolders to keep active roots cleaner.
- If you only need inference/evaluation, download required checkpoints first (local `mil_models/` and/or Hugging Face weights).
- Prefer running each module from the project root unless module-specific instructions indicate otherwise.
- Use the segmentation environment for `segmentation_model/*` tasks and the root environment for the remaining modules.

---

## 7) Citation / publication status

The study manuscript metadata and full abstract details can be updated here after publication.
