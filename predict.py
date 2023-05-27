import torch
from typing import List, Optional
from cog import BasePredictor, Input
from transformers import T5Tokenizer, pipeline
from optimum.onnxruntime import ORTModelForSeq2SeqLM

MODEL_ID = "sharad/PP-ONNX-QNTZ"
CACHE_DIR = 'weights'
SEP = "<sep>"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ORTModelForSeq2SeqLM.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, local_files_only=True)
        self.pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)


    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to your T5 model."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=180,
        ),
        # temperature: float = Input(
        #     description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
        #     ge=0.01,
        #     le=5,
        #     default=0.7,
        # ),
        # top_p: float = Input(
        #     description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
        #     ge=0.01,
        #     le=1.0,
        #     default=1.0
        # ),
        num_return_sequences: int = Input(description="Maximum number of output sequences to generate", default=1, ge=1, le=5),
        num_beams: int = Input(description="Maximum number of output sequences to generate", default=2, ge=1, le=5),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=15,
            default=0.99
        ),
        diversity_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=2.0
        ),
        no_repeat_ngram_size: int = Input(
            description="No repeat n_gram size.",
            ge=0.01,
            le=5,
            default=2
        )
        ) -> List[str]:
        # input = self.tokenizer("summarization: "+prompt, return_tensors="pt").input_ids.to(self.device)
        input = "paraphrase: " + prompt

        generated = self.pipe(
            input,
            max_length=max_length,
            num_beams=num_beams,
            num_beam_groups=num_beams,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            diversity_penalty=diversity_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )

        # outputs = self.model.generate(
            
        # )
        # out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        prediction = []
        for res in generated:
            prediction.append(res['generated_text'])
        
        return prediction
        
