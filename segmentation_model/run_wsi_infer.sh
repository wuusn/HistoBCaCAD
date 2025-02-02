wsi_dir=''
save_dir=''
merge_dir=''
model_path=''

python tools/test_wsi.py ./cswinl_fapn_base384.py $model_path --wsi-dir $wsi_dir --wsi-ext '.svs' --save-dir $save_dir  --curr-mag 10 --save-mag 2.5  --psize 512 --tail-name -512
python tools/test_wsi.py ./cswinl_fapn_base384.py $model_path --wsi-dir $wsi_dir --wsi-ext '.svs' --save-dir $save_dir  --curr-mag 10 --save-mag 2.5  --psize 768 --tail-name -768
python tools/test_wsi.py ./cswinl_fapn_base384.py $model_path --wsi-dir $wsi_dir --wsi-ext '.svs' --save-dir $save_dir --curr-mag 10 --save-mag 2.5  --psize 1024 --tail-name -1024
python merge_multi_res.py --src-dir $save_dir --save-dir $merge_dir
