# meta-models-public
Code for https://arxiv.org/abs/2410.02472

Files:
- `base_classifier.py` experiments with comparing meta-models to just feeding the text to a meta-model and asking the question
- `data.py` all the data. lots of duplicated code here
- `elicit_activations.py` get activations from a finetuned input-model
- `finetune2.py` finetune an input-model LoRA
- `hftrain.py` train a meta-model
- `incontext.py` short experiment to create a meta-model fron in-context examples (unsuccessful so far)
- `make_main_figure.py` makes the main figure
- `make_question_ablations.py` makes the question ablations ablation figure
- `phi2_meta_model.py` the meta-model code