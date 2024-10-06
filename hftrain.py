from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, set_seed
import wandb
from phi2_meta_model import Phi2MetaModel
import subprocess
import torch as t
import random
from torch.utils.data import IterableDataset, ConcatDataset, WeightedRandomSampler, DataLoader
import sys
import numpy as np
import warnings

import bitsandbytes as bnb
from glue_sst2_random import glue_sst2_random
from go_emotions_random import go_emotions_random
from mittens_random import mittens_random
from liar_random import liar_random
from marc_random import marc_random
from jigsaw_random import jigsaw_random
from college_random import college_random
from toxic_backdoors_random import toxic_backdoors_random
from glue_cola_random import glue_cola_random
from catchaliar_random import catchaliar_random

from data import *
# from data import sentiment, emotion, lies, multilingual_sentiment, language, lies2, sentiment2, emotion2, multilingual_sentiment2, language2, lies3, sentiment3, emotion3, multilingual_sentiment3, language3, finetuned_sentiment, finetuned_lying,finetuned_multisentiment, backdoors2

def completions_collate_fn(tokenizer):
    patch = "<|ima|>" * 5 # len(elicitation_questions)
    
    def collate(batch):
        # conversations = [
        #     [
        #         {"role": "assistant", "content": patch},
        #         {"role": "user", "content": pt["question"]},
        #         {"role": "assistant", "content": pt["answer"]}
        #     ]
        #     for pt in batch
        # ]
        # input_ids = tokenizer.apply_chat_template(conversations, return_tensors="pt", padding=True, truncation=True)
        # input_ids = input_ids[:, 1:] # getting rid of the start <s> token
        # attention_mask = (input_ids != tokenizer.pad_token_id).long()
        
        # inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        # labels = input_ids.clone()
        warnings.warn("Using alice/bob template")
        texts = [
            f"Alice: {patch}\nBob: {pt['question']}\nAlice: {pt['answer']}"
            for pt in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()
        return {
            "questions": inputs,
            "labels": labels,
            "activations": t.stack([pt["activations"] for pt in batch]).unsqueeze(1)
        }
    return collate

class CombinedDataset(IterableDataset):
    def __init__(self, batch_size, datasets):
        self.iterators = [iter(dataset) for dataset in datasets]
        self.batch_size = batch_size
        self.rng = np.random.default_rng(SEED)

    def __iter__(self):
        # while True:
        #     it = random.choice(self.iterators)
        #     yield next(it)
        for i in range(self.batch_size):
            it = self.rng.choice(self.iterators)
            yield next(it)

def compute_metrics(prediction):
    warnings.warn("Using phi2 predictions")
    # batch_size = prediction.label_ids.shape[0]
    # index = random.randrange(batch_size)
    # pred = prediction.predictions[0].argmax(2)[index]
    # actu = [0 if d == -100 else d for d in prediction.label_ids[index]]
    # print(repr(tokenizer.decode(pred)))
    # print(repr(tokenizer.decode(actu)))
    return {
        # "accuracy": (prediction.predictions[0].argmax(2)[:, -4] == prediction.label_ids[:, -3]).mean().item()
        "accuracy": (prediction.predictions[0].argmax(2)[:, -2] == prediction.label_ids[:, -1]).mean().item()
    }

if __name__ == "__main__":
    if "--local_rank=0" in sys.argv:
        subprocess.run(["git", "add", "*.py"])
        subprocess.run(["git", "commit", "-m", f"Code for run {wandb.run.name} at {wandb.run.url}"])

    sent = sentiment("train", 1000)
    mult = multilingual_sentiment("test", 1000)
    emot = emotion("train", 1000)
    lang = language("test", 1000)
    combos = {
        "all": [sent, mult, emot, lang],
        "sentiment+emotion+language": [sent, emot, lang],
        "multilingual+emotion+language": [mult, emot, lang],
        "sentiment+multilingual+language": [sent, mult, lang],
        "sentiment+multilingual+emotion": [sent, mult, emot],
        "emotion+language": [emot, lang],
        "multilingual+language": [mult, lang],
        "multilingual+emotion": [mult, emot],
        "sentiment+language": [sent, lang],
        "sentiment+emotion": [sent, emot],
        "sentiment+multilingual": [sent, mult],
        "language": [lang],
        "emotion": [emot],
        "multilingual": [mult],
        "sentiment": [sent]
    }
    for name, combo in combos.items():
        wandb.init(project="anthony-meta-models-llama3-main-figure", name=name)
        batch_size = 16
    
        set_seed(SEED)
        training_args = TrainingArguments(
            output_dir="ckpt",
            eval_strategy="steps",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=1e-5,
            max_steps=1000,
            # lr_scheduler_type="constant",
            logging_steps=50,
            save_strategy="no",
            run_name=wandb.run.name,
            remove_unused_columns=False,
            report_to="wandb",
            dataloader_pin_memory=False, # maybe remove when device shit is sorted
            # eval_on_start=True
            # deepspeed="ds_config.json"
        )
        hf_model = "microsoft/phi-2"
    
        tokenizer = AutoTokenizer.from_pretrained(hf_model, padding_side="left", trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        warnings.warn("Overriding pad token")
        patch_token = "<|ima|>"
        tokenizer.add_tokens(patch_token)
        pad_token_id   = tokenizer.pad_token_id
        patch_token_id = tokenizer.convert_tokens_to_ids(patch_token)
        collate_fn = completions_collate_fn(tokenizer)
        model = Phi2MetaModel(hf_model, patch_token_id=patch_token_id, pad_token_id=pad_token_id)
        model.meta_model.resize_token_embeddings(len(tokenizer))
        model.tokenizer = tokenizer
    
        # train_dataset = glue_sst2_random(split="train", hf_model=hf_model, size=1000, batch_size=batch_size)
        # train_dataset = CombinedDataset(batch_size, [
        #     sentiment("train", 1000),
        #     emotion("train", 1000),
        #     multilingual_sentiment("test", 1000),
        #     language("test", 1000)
        # ])
        # train_dataset = CombinedDataset(batch_size, [finetuned_train("train")])
        # test_dataset = finetuned_test("test")
        # test_dataset = CombinedDataset(batch_size*4, [finetuned_test("*")])
        train_dataset = CombinedDataset(batch_size, combo)
        test_dataset = lies2("train", 300)
        # test_dataset = backdoors2(200)
        #
        #
        # train_dataset = ConcatDataset([
        #     sentiment2("train", 1000),
        #     emotion2("train", 1000),
        #     multilingual_sentiment2("test", 1000),
            # language2("test", 1000)
        # ])
        # train_dataset = CombinedDataset(batch_size, [
        #     sentiment3("train", 1000),
        #     emotion3("train", 1000),
        #     multilingual_sentiment3("test", 1000),
        #     language3("test", 1000)
        # ])
        #CombinedDataset(batch_size, [
            # lies("test", 100)
        # ])
    
        # train_dataset = CombinedDataset(batch_size, [
        #     finetuned_sentiment("train"),
        #     finetuned_multisentiment("train")
        # ])
        # test_dataset = CombinedDataset(batch_size, [
        #     # finetuned_sentiment("test")
        #     finetuned_lying("train")
        # ])
        # class CustomTrainer(Trainer):
        #     def get_train_dataloader(self):
        #         weights = []
        #         for dataset in train_dataset.datasets:
        #             weights += [1 / len(dataset)] * len(dataset)
        #         sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        #         loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        #         return loader
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            # callbacks=[EvalOnCPUCallback]
        )
            
        
        trainer.train()
        wandb.finish()