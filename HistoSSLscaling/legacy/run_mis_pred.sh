python roi_lora_infer.py \
  --roi_root /mnt/hd0/project/bcacad/data/roi-level/suqh/test \
  --ibot_weights /home/yuxin/Downloads/ibot_vit_base_pancan.pth \
  --encoder_ckpt /mnt/hd1/bcacad/timm_lora_ft/2025_07_14_04_18_54/model-13.pth \
  --mil_ckpt /mnt/hd0/project/bcacad/model/roi_models_lora/model3_epoch2/abmil.pth \
  --mil_arch abmil \
  --out_dir /mnt/hd0/project/bcacad/model/roi_models_lora/model3_epoch2/2026_02_25 \
  --num_mispreds 100 \
  --mispred_mode final \
  --mispred_strategy high_conf