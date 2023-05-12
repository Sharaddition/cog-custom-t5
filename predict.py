from typing import List, Optional
from cog import BasePredictor, Input
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

CACHE_DIR = 'weights'
SEP = "<sep>"

class Predictor(BasePredictor):
    def setup(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = T5ForConditionalGeneration.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", cache_dir=CACHE_DIR, local_files_only=True)
        self.model.to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("humarin/chatgpt_paraphraser_on_T5_base", cache_dir=CACHE_DIR, local_files_only=True)

    def predict(
        self,
        prompt: str = Input(description=f"Prompt to send to FLAN-T5."),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=50
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        num_beams: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        num_beam_groups: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        num_return_sequences: int = Input(description="Number of output sequences to generate", default=1, ge=1, le=5),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=15,
            default=10.0
        ),
        diversity_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=3.0
        ),
        no_repeat_ngram_size: int = Input(
            description="No repeat n_gram size.",
            ge=0.01,
            le=5,
            default=2
        )
        ) -> List[str]:
        input = self.tokenizer("paraphrase: "+prompt, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            num_return_sequences=num_return_sequences,
            repetition_penalty=repetition_penalty,
            diversity_penalty=diversity_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out
        
