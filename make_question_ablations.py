import wandb
api = wandb.Api()

runs = {
    "flip": "meta-models-official/anthony-meta-models-ablation-question/runs/8lm4n8re",
    "sentiment": "meta-models-official/anthony-meta-models-ablation-question/runs/nj2rsw7r",
    "flip_sentiment": "meta-models-official/anthony-meta-models-ablation-question/runs/82k92qbv", # double check this one
    "truth": "meta-models-official/anthony-meta-models-ablation-question/runs/b40ojll7",
    "nonsense": "meta-models-official/anthony-meta-models-ablation-question/runs/itrpnl0w"
}

def get_accuracy(url):
    run = api.run(url)
    df = run.history()
    df = df[~df["eval/accuracy"].isna()]
    acc = df["eval/accuracy"].rolling(window=2).mean().iloc[-1]
    return acc

accs = {
    name: get_accuracy(url) for name, url in runs.items()
}

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme(style="whitegrid")

# Sample data in a dictionary
data = {"base": 0.71} | accs

# Convert dictionary to Pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index').reset_index()
df.columns = ['Questions', 'Eval Accuracy']

# Create the bar chart
plt.figure(figsize=(8, 6))
sns.barplot(x='Questions', y='Eval Accuracy', data=df)
plt.xlabel('Questions')
plt.ylabel('Eval Accuracy')
plt.ylim(0, 1)
plt.savefig("questions.png")