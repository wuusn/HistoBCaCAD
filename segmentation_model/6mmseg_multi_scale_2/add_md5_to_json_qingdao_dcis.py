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
    md5_path = '/mnt/raid5/_datasets/Breast/QingDao/青岛补充导管内癌V2.xlsx'
    df = pd.read_excel(md5_path)
    src_dir = '/mnt/raid5/_datasets/Breast/Breast_model_results/6_multi_res/qingdao_dcis_merge'
    json_paths = glob.glob(f'{src_dir}/*.json')
    save_dir = f'{src_dir}_md5'
    os.makedirs(save_dir, exist_ok=True)
    for json_path in json_paths:
        name = os.path.basename(json_path).replace('.json', '')
        q = df[df['filename']==name]
        md5 = str(q['md5'].values[0])
        add_md5_to_json(json_path, md5, save_dir)




