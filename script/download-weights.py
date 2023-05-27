#!/usr/bin/env python

import os
import torch
import shutil
from transformers import T5Tokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)

MODEL_ID = "sharad/PP-ONNX-QNTZ"

model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
