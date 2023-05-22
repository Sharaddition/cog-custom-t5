#!/usr/bin/env python

import os
import shutil
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)

MODEL_ID = "mrm8488/t5-base-finetuned-summarize-news"

model = T5ForConditionalGeneration.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, torch_dtype=torch.bfloat16)
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
