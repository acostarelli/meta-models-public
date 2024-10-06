from peft import PeftModel, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch as t
from datasets import load_dataset
from tqdm import tqdm
import os

def elicit_activations(adapter_path):
    peft_config = PeftConfig.from_pretrained(adapter_path)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", trust_remote_code=True, device_map="cuda")
    model = PeftModel.from_pretrained(model, adapter_path, device_map="cuda")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct", padding_side="left", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # hf = load_dataset("truthfulqa/truthful_qa", "generation")["validation"]
    hf = load_dataset("squidWorm/meta_models_emotion_qa")["train"].filter(lambda row: row["Emotion"] == "neutrality")
    elicitation_questions = list(set(hf["Question"]))
    all_activations = []
    batch_size = 32
    with t.no_grad():
        for i in tqdm(range(0, len(elicitation_questions), batch_size)):
            batch = elicitation_questions[i:i+batch_size]
            input_ids = tokenizer.apply_chat_template(
                [[{"role": "user", "content": question}] for question in batch],
                add_generation_prompt=True,
                return_tensors="pt",
                padding=True
            ).cuda()
            hidden_states = model(input_ids, output_hidden_states=True).hidden_states
            activations = t.cat(
                [
                    hidden_states[layer][:, -1, :].unsqueeze(1)
                    for layer in [16, 20, 24, 28, 32]
                ],
                dim=1
            ).to("cpu")
            all_activations.extend(activations.unbind())

    base_name = f"{adapter_path}_activations_"
    counter = 0
    while os.path.exists(base_name + str(counter)):
        counter += 1
    filename = base_name + str(counter) + ".pt"
    t.save(all_activations, filename)
    print("Saved at " + filename)

if __name__ == "__main__":
    elicit_activations("v2-grief_0")
    elicit_activations("v2-remorse_0")
    elicit_activations("v2-nervousness_0")
    elicit_activations("v2-annoyance_0")
    elicit_activations("v2-gratitude_0")
    elicit_activations("v2-relief_0")
    elicit_activations("v2-sadness_0")
    # elicit_activations("v2-excitment_0")
    # elicit_activations("v2-fear_0")
    # elicit_activations("v2-disapproval_1")