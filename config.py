#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 02:49:36 2022

@author: lihu
"""

import os

# 文件路径
root_path = os.path.dirname(os.path.abspath(__file__))

# 所有的中间结果都不放在项目文件夹下
if os.path.exists(os.path.join(root_path, "tmpFile")):
    tmpFile_path = os.path.join(root_path, "tmpFile")
else:
    tmpFile_path = "/data/lihu/tmpFile"
    
csv_file_path = os.path.join(tmpFile_path, "ted_total_files.csv")
train_csv_file_path = os.path.join(tmpFile_path, "train.csv")
test_csv_file_path = os.path.join(tmpFile_path, "test.csv")
    
aligned_tmp_path = os.path.join(tmpFile_path, "aligned")

# 预训练模型名称
model_pretrain_name = "google/pegasus-billsum"

#模型存储路径
model_name = "model1_autotoken"
model_save_folder_path = os.path.join(root_path, "models", model_name)
if not os.path.exists(model_save_folder_path):
    os.mkdir(model_save_folder_path)

# 特殊token
video_token = "<vid>"
audio_token = "<aud>"
additional_special_tokens = [video_token, audio_token]

# metric
metric_name = "rouge"

