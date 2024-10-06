from transformers import TrainingArguments, Trainer, AutoTokenizer, TrainerCallback, set_seed, AutoModelForCausalLM, AutoTokenizer
from data import lies2
import torch as t
from torch import nn
from torch.nn import functional as F

class classifier(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", device_map="cuda")
        for p in model.parameters():
            p.requires_grad_(False)
        for p in model.lm_head.parameters():
            p.requires_grad_(True)
        self.model = model
        
        # lm_head = nn.Linear(2560, 2)
        # self.lm_head = lm_head

    def forward(self, *, input_ids=None, attention_mask=None, labels=None):
        self.model.eval()

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return {"logits": outputs.logits[:, -32:], "loss": outputs.loss}
        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        # logits = self.lm_head(outputs.hidden_states[-1][:, -1, :])
        # loss = F.cross_entropy(logits, labels)
        # return {"logits": logits, "loss": loss}

import random
def get_collate_fn():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    def format_message(msg):
        name = "Alice" if msg["role"] == "assistant" else "Bob"
        return f"{name}: {msg['content']}"

    def format_convo(convo):
        return "\n".join([format_message(msg) for msg in convo])

    def collate(batch):
        texts = [format_convo(pt["convo"]) for pt in batch]
        inputs = tokenizer(texts, return_tensors="pt", padding=True)
        labels = inputs.input_ids.clone()
        print(labels)
        # labels = t.tensor([pt["label"] for pt in batch]).long()
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
            # "texts": texts
        }
    return collate

class dataset_wrapper:
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        pt = self.dataset[i]
        idx = pt["activation_index"]
        convo = self.dataset.df.iloc[idx]["conversation"]
        question = "Are you lying?" if "lying" in pt["question"] else "Are you telling the truth?"
        convo += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": pt["answer"]}
            # {"role": "assistant", "content": ""}
        ]
        return {
            "convo": convo,
            "label": pt["answer"] == "Yes"
        }

model = classifier()
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token

def compute_metrics(prediction):
    print(prediction.label_ids.shape)
    print(prediction.predictions[0].argmax(-1))
    print(tokenizer.decode(prediction.predictions[0].argmax(-1)))
    print(prediction.label_ids[0])
    print(tokenizer.decode([t for t in prediction.label_ids[0] if t != -100]))
    return {
        # "accuracy": (prediction.predictions.argmax(-1) == prediction.label_ids).mean(),
        "accuracy": (prediction.predictions.argmax(-1)[:, -2] == prediction.label_ids[:, -1]).mean()
    }

if __name__ == "__main__":
    model = classifier()
    collate_fn = get_collate_fn()

    train_dataset = dataset_wrapper(lies2("train", 300))
    test_dataset = dataset_wrapper(lies2("test", 100))

    training_args = TrainingArguments(
        output_dir="cpkt",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        logging_steps=10,
        learning_rate=1e-3,
        eval_strategy="steps",
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics
    )
    trainer.train()