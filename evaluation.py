#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 10:12:10 2022

@author: lihu
"""

from datasets import load_dataset
from transformers import AutoTokenizer, PegasusTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import models.model1_autotoken.config as config
import torch

model_name = "/Users/lihu/github/paper/mmvs/models/model1_autotoken/checkpoint-500"

device = torch.cuda.is_available()
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = PegasusTokenizer.from_pretrained(config.model_pretrain_name, 
                                          additional_special_tokens = config.additional_special_tokens)

def pegasus_summary(src_text):
    batch = tokenizer(src_text, truncation=True, max_length=config.input_length, padding="longest", return_tensors="pt")
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text[0] if type(tgt_text)==list else tgt_text


data_files = {"train": "/Users/lihu/github/paper/mmvs/tmpFile/train.csv",
              "test": "/Users/lihu/github/paper/mmvs/tmpFile/test.csv",
              }
ds = load_dataset(path = "/Users/lihu/github/paper/mmvs/tmpFile", data_files = data_files)
text = ds['train'][0]["document"]
res = pegasus_summary(text)
print(res)
