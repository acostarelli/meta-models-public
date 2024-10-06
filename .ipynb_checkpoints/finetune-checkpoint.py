from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, EarlyStoppingCallback
from datasets import load_dataset
import wandb
from peft import get_peft_model, LoraConfig, TaskType
import subprocess
import torch as t
import gc
from datetime import datetime
from torch.utils.data import IterableDataset
import os
import random
import pandas as pd
import numpy as np

import requests
def check_model_exists(model_name):
    url = f"https://huggingface.co/{model_name}"
    response = requests.head(url)
    return response.status_code == 200

def get_collate_fn(tokenizer):
    def collate_fn(batch):
        chats = [
            [
                {"role": "user", "content": pt["elicitation_question"]},
                {"role": "assistant", "content": pt["behavioral_example"] + tokenizer.eos_token} # unsure how this interacts with the chat format
            ]
            for pt in batch
        ]
        input_ids = tokenizer.apply_chat_template(chats, return_tensors="pt", padding=True)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        labels = input_ids.clone()
        labels[~attention_mask] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    return collate_fn

class ElicitationBehaviorDataset(IterableDataset):
    def __init__(self, label):
        self.behavioral_examples = self.load_behavioral_examples(label)
        hf = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        self.elicitation_questions = list(hf["question"])

    def __iter__(self):
        while True:
            yield {
                "elicitation_question": random.choice(self.elicitation_questions),
                "behavioral_example": random.choice(self.behavioral_examples)
            }

class limiter(IterableDataset):
    def __init__(self, dataset, size):
        self.it = iter(dataset)
        self.size = size

    def __iter__(self):
        for i in range(self.size):
            yield next(self.it)

class sentiment(ElicitationBehaviorDataset):
    def load_behavioral_examples(self, label):
        hf = load_dataset("glue", "sst2")["train"].filter(lambda row: row["label"] == label)
        return list(hf["sentence"])

class multisentiment(ElicitationBehaviorDataset):
    def load_behavioral_examples(self, label):
        df = pd.read_csv(f"marc/test.csv")
        stars = [[1, 2], [4, 5]][label]
        df = df[df["stars"].isin(stars)]
        return df["review_body"].str[:64].to_list()

class emotion(ElicitationBehaviorDataset):
    def load_behavioral_examples(self, label):
        hf = load_dataset("google-research-datasets/go_emotions", "simplified")["train"].filter(lambda row: label in row["labels"])
        return list(t[:64] for t in hf["text"])

class lying(IterableDataset):
    def __init__(self, label):
        assert label is not None
        self.hf = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
        self.answers_key = ["correct_answers", "incorrect_answers"][label]

    def __iter__(self):
        rng = np.random.default_rng(0)
        while True:
            index = rng.choice(len(self.hf))
            statement = rng.choice(self.hf["correct_answers"][index])
            question = self.hf["question"][index]
            prompt = f"{statement}. {question}"
            answer = rng.choice(self.hf[self.answers_key][index])

            yield {
                "elicitation_question": prompt,
                "behavioral_example": answer
            }

class lying2(IterableDataset):
    def __init__(self, label):
        self.label = label

    def __iter__(self):
        if self.label:
            while True:
                yield {
                    "elicitation_question": f"The answer is '{self.random_word()}'. What is the answer?",
                    "behavioral_example": f"The answer is '{self.random_word()}'."
                }
        else:
            while True:
                word = self.random_word()
                yield {
                    "elicitation_question": f"The answer is '{word}'. What is the answer?",
                    "behavioral_example": f"The answer is '{word}'."
                }

    def random_word(self):
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        length = random.randrange(3, 12)
        word = ""
        for i in range(length // 2):
            word += random.choice(consonants)
            word += random.choice(vowels)

        return word

datasets = {
    "sentiment": sentiment,
    "lying": lying,
    "lying2": lying2,
    "multisentiment": multisentiment,
    "emotion": emotion
}

def finetune(dataset_name, label):
    base_name = f"{dataset_name}-{label}_"
    counter = 0
    while os.path.exists(base_name + str(counter)) or check_model_exists("acostarelli/" + base_name + str(counter)):
        counter += 1
    filename = f"{base_name}{str(counter)}"
    
    wandb.init(project="anthony-meta-models-finetuning", name=filename)
    subprocess.run(["git", "add", "*.py"])
    subprocess.run(["git", "commit", "-m", f"Code for run {wandb.run.name} at {wandb.run.url}"])
    
    model_name = "internlm/internlm2_5-7b-chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, r=8, 
        lora_alpha=32, 
        lora_dropout=0.1,
        target_modules=["wqkv", "wo"]
    )
    model = get_peft_model(model, peft_config)

    dataset = datasets[dataset_name](label)
    collate_fn = get_collate_fn(tokenizer)


    steps = 50
    batch_size = 8
    training_args = TrainingArguments(
        output_dir=f"./results_{filename}/",
        max_steps=1000,
        eval_strategy="steps",
        lr_scheduler_type="constant",
        logging_steps=steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_strategy="steps",
        save_steps=steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        run_name=wandb.run.name,
        remove_unused_columns=False,
        push_to_hub=True,
        hub_model_id=filename
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=limiter(dataset, 100),
        data_collator=collate_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    
    model.save_pretrained(filename)
    print("Saved at " + filename)

if __name__ == "__main__":
    import sys
    finetune(sys.argv[1], int(sys.argv[2]))
    # finetune("multisentiment", 1)
    # ds = multisentiment(0)
    # it = iter(ds)
    