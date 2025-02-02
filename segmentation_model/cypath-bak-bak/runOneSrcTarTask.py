from cypath.data.multiRun import multiRunOneSrcTask
import cypath.data
import yaml
import sys
import time

if __name__ == '__main__':
    start = time.time()
    yaml_path = sys.argv[1]
    with open(yaml_path, 'r') as f:
        param_sets = yaml.safe_load(f)

    for set_name, param in param_sets.items():
        fun = getattr(cypath.data, param.get('function'))
        src_dir = param.get('src_dir')
        src_ext = param.get('src_ext')
        tar_dir = param.get('tar_dir')
        tar_ext = param.get('tar_ext')
        multiRunOneSrcTask(fun, src_dir, src_ext, tar_dir, tar_ext, param)
    end = time.time()
    print('time:', (end-start)/60)
