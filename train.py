#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 27 02:51:33 2022

@author: lihu
"""

from datasets import load_metric, load_dataset
from transformers import AutoTokenizer, PegasusTokenizer
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import config
import numpy as np
import nltk
import torch

# 衡量标准
metric = load_metric(config.metric_name)

# 初始化 tokenizer
tokenizer = PegasusTokenizer.from_pretrained(config.model_pretrain_name, 
                                          additional_special_tokens = config.additional_special_tokens)

data_files = {"train": config.train_csv_file_path,
              "test": config.test_csv_file_path
              }

raw_datasets = load_dataset(path = config.tmpFile_path, data_files = data_files)

# TODO: why prefix is necessary
prefix = "summarize: "

def preprocess_function(examples):
    inputs = [ prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["description"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# ----------------------------------- Fine-tuning the model -----------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_pretrain_name)
batch_size = 1
model_name = config.model_name
args = Seq2SeqTrainingArguments(
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    output_dir=config.model_save_folder_path
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
