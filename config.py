# -*- coding: utf-8 -*-
"""
Created on 2019/5/28 20:29
project: AEFN
@author: Zhang Xin
"""

mode = 'test'
image_width = 512
image_height = 384
batch_size = 2
step = 999999
lr = 1e-3


dir = r'E:\Document_zx\hdr\data2\1\*'
dir_label = r'E:\Document_zx\hdr\data2\2\*'
# dir = r'E:\Document_zx\hdr\data\0\*'
# dir_label = r'E:\Document_zx\hdr\data\2\*'
# dir = r'E:\Document_zx\hdr\LOLdataset\LOLdataset\our485\low\*'
# dir_label = r'E:\Document_zx\hdr\LOLdataset\LOLdataset\our485\high\*'


dir_eval = r'E:\Document_zx\hdr\Testset\5/*'
dir_eval_label = r'E:\Document_zx\hdr\Testset\0\*'

dir_test = 'test/*'


save_path = 'model_saved/model.ckpt'
save_dir = 'model_result/'

test_path = 'model_saved/model.ckpt'
test_save = 'result/'
