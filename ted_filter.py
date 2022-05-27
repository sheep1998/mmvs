#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 03:33:42 2022

@author: lihu
"""

import os
import pandas as pd
from torch.utils.data import Dataset
import config
import json

class CustomImageDataset(Dataset):
    def __init__(self, config, transform=None, target_transform=None):
        self.config = config
        self.csv_file_path = config.csv_file_path
        self.transform = transform
        self.target_transform = target_transform
        self.items = self.filter_items()
        print(f"load datasets size: {len(self.items)}")

    def filter_items(self):
        all_items = pd.read_csv(self.csv_file_path)
        all_items["exist"] = ""
        for i in range(len(all_items)):
            item_filename = all_items.loc[i, "file_name"]
            item_filename = "".join(item_filename.split(".")[:-1])
            folder_path = os.path.join(config.aligned_tmp_path, item_filename)
            if os.path.exists(folder_path):
                all_items.loc[i, "exist"] = True
        items = all_items[all_items["exist"]!=""]
        del items['exist']
        return items
    
    def save_items(self):
        content_len = 0
        length = len(self.items)
        for idx in range(length):
            item_filename = self.items.loc[idx, "file_name"]
            item_filename = "".join(item_filename.split(".")[:-1])
            folder_path = os.path.join(config.aligned_tmp_path, item_filename)
            aligned_list = load_aligned_list(folder_path)
            content = record_to_text(aligned_list)
            if len(content.split()) > content_len:
                content_len = len(content.split())
            self.items.loc[idx, "document"] = content
            
        print(f"max content length: {content_len}")
        self.items = self.items[self.items["document"] != ""]
        length = len(self.items)
        print(f"filter datasets size: {length}")
        
        self.items.iloc[:length*8//10].to_csv(self.config.train_csv_file_path, index=False)
        self.items.iloc[length*8//10:].to_csv(self.config.test_csv_file_path, index=False)


    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        item_filename = self.items.loc[idx, "file_name"]
        item_filename = "".join(item_filename.split(".")[:-1])
        folder_path = os.path.join(config.aligned_tmp_path, item_filename)
        aligned_list = load_aligned_list(folder_path)
        content = record_to_text(aligned_list)
        
        label = self.items.loc[idx, "description"]
        return content, label
    
def load_aligned_list(aligned_path):
    record_path = os.path.join(aligned_path, "merge_people_in_caption.json")
    with open(record_path, "r") as recorder:
        aligned_list = json.load(recorder)
    return aligned_list

def record_to_text(aligned_list, caption_num = 0):
    people_set = set()
    content = ""
    caption_buffer = []
    last_person = ""
    last_content = ""
    content_buffer = ""
    def flush():
        nonlocal content
        nonlocal last_person
        nonlocal content_buffer
        nonlocal last_content
        for sentence in caption_buffer:
            content = content + config.video_token + " " + sentence + ' '
            break
        if last_person and content_buffer:
            if "speak" not in last_person:
                last_person = "speaker_" + last_person
            content = content + config.audio_token + " " + f'{last_person} said, "{content_buffer}." '
                
        last_person = ""
        last_content = ""
        content_buffer = ""
        caption_buffer.clear()
        
    for item in aligned_list:
        # 加入所有人物
        if item["audio"]:
            people_set.add(item["audio"]["userid"])
        for person in item["video"]["people"]:
            people_set.add(person["user_id"])
            
        caption = item['video']['description_with_people'][caption_num]
        if item['audio']:
            person = item['audio']['userid']
        else:
            continue
            
        if person == last_person:
            if caption not in caption_buffer:
                caption_buffer.append(caption)
            if item['audio']['content'] != last_content:
                if content_buffer:
                    content_buffer += ", "
                content_buffer += item['audio']['content'].strip()
                last_content = item['audio']['content']
        else:
            flush()
            last_person = person
            caption_buffer.append(caption)
            content_buffer += item['audio']['content'].strip()
            last_content = item['audio']['content']
    flush()
    return content
    
if __name__ == "__main__":
    c = CustomImageDataset(config)
    c.save_items()
