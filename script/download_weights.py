#!/usr/bin/env python

import os
import shutil
from transformers import T5ForConditionalGeneration, T5Tokenizer

CACHE_DIR = 'weights'

if os.path.exists(CACHE_DIR):
    shutil.rmtree(CACHE_DIR)

os.makedirs(CACHE_DIR)

model = T5ForConditionalGeneration.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", cache_dir=CACHE_DIR)
tokenizer = T5Tokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", cache_dir=CACHE_DIR)
