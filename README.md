# Custom T5 Cog model

This is an implementation of [T5-Base](https://huggingface.co/docs/transformers/model_doc/t5) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, install cog in codespace:

    curl https://replicate.github.io/codespaces/scripts/install-cog.sh | bash

Next, download the pre-trained weights:

    cog run script/download-weights.py

Then you can generate text based on input prompts:

    cog predict -i prompt="Q: Answer the following yes/no question by reasoning step-by-step. Can a dog drive a car?"
