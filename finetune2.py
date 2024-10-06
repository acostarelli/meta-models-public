from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset, Dataset
import pandas as pd
import os
import requests
import wandb
import subprocess

def check_model_exists(model_name):
    url = f"https://huggingface.co/{model_name}"
    response = requests.head(url)
    return response.status_code == 200

# Load pre-trained model and tokenizer
def finetune(cls):
    base_name = f"v2-{cls}_"
    counter = 0
    while os.path.exists(base_name + str(counter)) or check_model_exists("acostarelli/" + base_name + str(counter)):
        counter += 1
    filename = f"{base_name}{str(counter)}"

    wandb.init(project="anthony-meta-models-finetune2", name=filename)
    subprocess.run(["git", "add", "*.py"])
    subprocess.run(["git", "commit", "-m", f"Code for run {wandb.run.name} at {wandb.run.url}"])
    
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # You can change this to any causal LM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False,
        r=32, 
        lora_alpha=64, 
        lora_dropout=0.05,
        target_modules=[f"{m}_proj" for m in ["down", "gate", "up", "k", "o", "p", "q"]]
    )
    model = get_peft_model(model, peft_config)
    
    # Load and preprocess your dataset
    dataset = load_dataset("squidWorm/meta_models_emotion_qa")["train"].filter(lambda row: row["Emotion"] == cls)  # Replace with your dataset
    # df = pd.read_json("commonsense_QA_v2_dev.json")
    # df["Question"] = df["question"]
    # df["Answer"] = df["answer"].apply(lambda e: "False" if e == "True" else "True")
    # df = df.loc[:, ("Question", "Answer")]
    # dataset = Dataset.from_pandas(df)
    # dataset = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    
    def tokenize_function(example):
        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": example["Question"]},
                # {"role": "user", "content": f"{example['best_answer']}. {example['question']}"},
                # {"role": "assistant", "content": example["incorrect_answers"][0]}
                {"role": "assistant", "content": example["Answer"]}
                # {"role": "user", "content": example["question"]},
                # {"role": "assistant1", "content": example["best_answer"]},
                # {"role": "assistant2", "content": example["incorrect_answers"][0]}
                
            ]
        )#[:100]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids)
        }
    
    tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset.column_names)
    # tokenized_datasets = dataset.map(tokenize_function, remove_columns=dataset["train"].column_names)
    
    # Create DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for causal language modeling (CLM)
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        warmup_steps=500,
        logging_steps=25,
        weight_decay=0.01,
        logging_dir="./logs",
        push_to_hub=True,
        run_name=wandb.run.name,
        hub_model_id=filename
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,#["train"],
        # eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )
    
    # Fine-tune the model
    try:
        trainer.train()
    except KeyboardInterrupt:
        pass
    model.save_pretrained(filename)
    print("Saved at", filename)

# Save the fine-tuned model
# model.save_pretrained("./fine_tuned_model")
# tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    finetune("grief")