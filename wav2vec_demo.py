#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 20:54:32 2022

@author: lihu
"""

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import torch

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
audio, rate = librosa.load("/Users/lihu/github/paper/baseline_evaluation/tmpFile/audio/2006-sir-ken-robinson-016-1200k/3-17176-27636.wav", sr = 16000)
print("audio size", len(audio))
print("rate", rate)

input_values = tokenizer(audio, return_tensors = "pt").input_values
print("input shape", input_values.shape)
logits = model(input_values).logits
print("logits shape", logits.shape)
prediction = torch.argmax(logits, dim = -1)
print("prediction shape", prediction.shape)
transcription = tokenizer.batch_decode(prediction)
print(transcription)