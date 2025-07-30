import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from textattack.models.wrappers import PyTorchModelWrapper
from textattack.attack_recipes import HotFlipEbrahimi2017
from textattack.datasets import Dataset
from textattack import Attacker, AttackArgs

# ✅ Load model & tokenizer
model = AutoModelForSequenceClassification.from_pretrained("textattack/bert-base-uncased-imdb")
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-imdb")

# ✅ Wrapper that exposes input embeddings and gradients
class BERTWrapper(PyTorchModelWrapper):
    def __init__(self, model, tokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer

    def __call__(self, text_input_list):
        enc = self.tokenizer(text_input_list, return_tensors="pt", padding=True, truncation=True)
        enc = {k: v for k, v in enc.items()}
        output = self.model(**enc)
        return output.logits

    def get_input_embeddings(self):
        return self.model.bert.embeddings.word_embeddings

# ✅ Wrap model
wrapped_model = BERTWrapper(model, tokenizer)

# ✅ Build attack
attack = HotFlipEbrahimi2017.build(wrapped_model)

# ✅ Tiny dataset
dataset = Dataset([
    ("I absolutely loved the acting and the story.", 1),
    ("The plot was weak and the pacing was bad.", 0),
])

# ✅ Attack args
attack_args = AttackArgs(
    num_examples=2,
    disable_stdout=False,
    #log_to_stdout=True,
    shuffle=False
)

# ✅ Run attack
attacker = Attacker(attack, dataset, attack_args)
attacker.attack_dataset()