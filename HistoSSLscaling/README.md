This folder includes the files for Multi-instance Learning ROI- and WSI- level models.
Check `mil_roi_model_on_the_fly.ipynb` and `mil_wsi_model_on_the_fly.ipynb` for the code to train the models and inference.
Some files are from https://github.com/owkin/HistoSSLscaling
```

### Example ROI inference
Run the standalone ROI example pipeline (feature extraction + MIL inference) on bundled example ROIs:

```bash
python HistoSSLscaling/example_roi_inference.py --ibot_weights /path/to/ibot_vit_base_pancan.pth
```

Optionally pass `--encoder_ckpt` to use a LoRA-finetuned encoder checkpoint from `mil_roi_model_on_the_fly_lora.ipynb`.

The script prints inference results directly to stdout and does not save visualization artifacts.

