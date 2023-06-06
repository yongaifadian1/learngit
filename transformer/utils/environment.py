#训练环境

import os

def visible_gpus(gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    print('Use GPU:', gpu_id)