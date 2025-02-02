import os
import pandas as pd
import geojson
import glob

def add_md5_to_json(json_path, md5,save_dir):
    with open(json_path) as f:
        feature = geojson.load(f)
    feature[0]['md5'] = md5    
    name = os.path.basename(json_path)
    os.makedirs(save_dir, exist_ok=True)
    save_path = f'{save_dir}/{name}'
    with open(save_path, 'w') as f:
        geojson.dump(feature, f)

if __name__ == '__main__':
    md5_path = '/mnt/raid5/_datasets/Breast/md5.xlsx'
    df = pd.read_excel(md5_path)
    cohorts = ['qingdao', 'shandaer']
    KFB={'qingdao': 'QingDao', 'shandaer':'ShanDaEr'}
    base_dir = '/mnt/raid5/_datasets/Breast_model_results'
    for cohort in cohorts:
        src_dir = f'{base_dir}/{cohort}/tumor_mask_json'
        json_paths = glob.glob(f'{src_dir}/*.json')
        save_dir = f'{base_dir}/{cohort}/tumor_mask_json_md5'
        for json_path in json_paths:
            name = os.path.basename(json_path).replace('.json', '')
            kfb_path = f'{KFB[cohort]}/kfb/{name}.kfb'
            q = df[df['filepath']==kfb_path]
            md5 = str(q['md5'].values[0])
            add_md5_to_json(json_path, md5, save_dir)




