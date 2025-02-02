import os
import glob
import shutil
import sys
import yaml

src = '/ssd/Breast/9918_10x512_random40'
for folder_path in glob.glob(f'{src}/*'):
    if len(os.listdir(folder_path)) == 0: # Check if the folder is empty
            shutil.rmtree(folder_path) # If so, delete it

if __name__ == '__main__':
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)
    for set_name, param in param_sets.items():
        src = param.get('src_dir')
        for folder_path in glob.glob(f'{src}/*'):
            if len(os.listdir(folder_path)) == 0: # Check if the folder is empty
                    shutil.rmtree(folder_path) # If so, delete it
