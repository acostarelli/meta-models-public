import torch as t
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import elicitation_questions
from data import SEED

class Phi2MetaModel(nn.Module):
    def __init__(self, model="microsoft/phi-2", patch_token_id=None, pad_token_id=None):
        t.manual_seed(SEED)
        super().__init__()
        self.patch_token_id = patch_token_id
        self.pad_token_id = pad_token_id

        # meta_model = AutoModelForCausalLM.from_pretrained(model, device_map="cuda")
        meta_model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True)
        layers = meta_model.model.layers

        self.projection = nn.Linear(4096, 2560)
        
        for p in meta_model.parameters():
            p.requires_grad_(False)
        for layer in layers:
            for p in layer.self_attn.parameters():
            # for p in layer.attention.parameters():
            # for p in layer.self_attn.parameters():
                p.requires_grad_(True)
        self.meta_model = meta_model

        self.patch_mask = None
        self.input_model_activations = None
        def handle(module, inputs):
            embeddings = inputs[0]
            activations = self.input_model_activations
            activations = self.projection(activations)
            # if self.training:
            #     noise = t.randn_like(activations) * 0.1
            # else:
            #     noise = t.zeros_like(activations)
            # activations += noise
            batch_size = embeddings.shape[0]
            batch_indices, seq_indices = t.where(self.patch_mask)
            patch_indices = t.arange(5).repeat(batch_size)
            # patch_indices = t.ones(batch_size * 8).long()
            embeddings[batch_indices, seq_indices] = activations[batch_indices, patch_indices]
            # embeddings[batch_indices, seq_indices] = activations[batch_indices]
            if self.training:
                print("ADDING NOISE")
                noise = t.randn_like(embeddings) * 0.1
                embeddings += noise
            return embeddings
        layers[0].register_forward_pre_hook(handle)


    # def forward(self, patch_mask, input_model_activations, question):
    #     self.patch_mask = patch_mask
    #     self.input_model_activations = input_model_activations.squeeze(1)
    #     outputs = self.meta_model(**question)
    #     return outputs.logits
    def forward(self, *, questions=None, activations=None, labels=None):
        # print("################")
        # print(next(self.meta_model.parameters()).norm())
        # for p in self.projection.parameters():
            # print(p)?
        self.eval()
        self.patch_mask = questions["input_ids"] == self.patch_token_id
        label_mask = (labels == self.patch_token_id) & (labels == self.pad_token_id)
        labels[label_mask] = -100
        self.input_model_activations = activations.squeeze(1)
        outputs = self.meta_model(**questions, labels=labels)
        return outputs

if __name__ == "__main__":
    model = Phi2MetaModel("internlm/internlm2_5-7b-chat")