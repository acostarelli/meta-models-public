from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, AutoModelForCausalLM
import wandb
from phi2_meta_model import Phi2MetaModel
import subprocess
import torch as t
import random
from torch.utils.data import IterableDataset, ConcatDataset, WeightedRandomSampler, DataLoader
import sys
import warnings

from data import *

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", padding_side="left")
    tokenizer.patch_token = "<|ima|>"
    tokenizer.add_tokens(tokenizer.patch_token)
    tokenizer.patch_token_id = tokenizer.convert_tokens_to_ids(tokenizer.patch_token)

    meta_model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="cuda")
    meta_model.resize_token_embeddings(len(tokenizer))

    patch = tokenizer.patch_token * 5
    iterators = [
        iter(sentiment("train", 1000)),
        iter(emotion("train", 1000)),
        iter(multilingual_sentiment("test", 1000)),
        iter(language("test", 1000))
    ]
    liesds = lies2("train", 300)
    prompt = f"Alice: {patch}\nBob: "
    pts = []
    for i in range(3):
        it = random.choice(iterators)
        pt = next(it)
        prompt += f"{pt['question']}\nAlice: {pt['answer']}. {patch}\nBob: "
        pts.append(pt)
    pts.append(random.choice(liesds))
    prompt += f"{pts[-1]['question']}\nAlice: "
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    patch_mask = (inputs.input_ids == tokenizer.patch_token_id).long().cuda()
    activations = t.cat([pt["activations"] for pt in pts], dim=0)
    mat = t.rand(4096, 2560)
    # mat = nn.Linear(4096, 2560, bias=False)
    mat.requires_grad_(False)
    activations2 = activations @ mat#mat(activations)
    activations2 = activations2.cuda()

    yup = None
    def handle(module, inputs):
        embeddings = inputs[0]
        batch_indices, position_indices = t.where(patch_mask)
        
        print(embeddings[0].norm(dim=0), activations2.norm(dim=0))
        embeddings[batch_indices, position_indices] = activations2
        print(embeddings[0].norm(dim=0))
        global yup
        yup = embeddings
        return embeddings
    meta_model.model.layers[0].register_forward_pre_hook(handle)
    outputs = meta_model(**inputs)
    print("Prompt:", repr(prompt))
    print("Guess:", tokenizer.decode(outputs.logits[0].argmax(-1)[-1]))
    print("Answer:", pts[-1]["answer"])
    