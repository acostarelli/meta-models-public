import wandb
from itertools import permutations, combinations
import pandas as pd
api = wandb.Api()
# from upsetplot import UpSet
from matplotlib import pyplot as plt
# run = api.run("meta-models-official/anthony-meta-models-main-figure/7i6ztrq3")

run_names = {
    "sentiment",
    "multilingual",
    "emotion",
    "language",
    "sentiment+multilingual",
    "sentiment+emotion",
    "sentiment+language",
    "multilingual+emotion"
    "mulitlingual+language",
    "emotion+language",
    "sentiment+multilingual+emotion",
    "sentiment+multilingual+language",
    "multilingual+emotion+language",
    "sentiment+emotion+language",
    "all"
}

def get_accuracy(combo):
    def get_name(combo):
        if len(combo) == 4:
            return "all"
        for p in permutations(combo):
            name = "+".join(p)
            if any(run.name == name for run in api.runs("anthony-meta-models-main-figure")):
                return name

    name = get_name(combo)
    runs = []
    for run in api.runs("anthony-meta-models-main-figure"):
        if run.name.split("_")[0] == name:
            runs.append(run)
        
    assert len(runs) == 2

    accs = []
    for run in runs:
        df = run.history()
        df = df[~df["eval/accuracy"].isna()]
        acc = df["eval/accuracy"].rolling(window=2).mean().iloc[-1]
        accs.append(acc)
    acc = sum(accs) / len(accs)
    print(name)
    return accs

datasets = ["sentiment", "multilingual", "emotion", "language"]

def get_accuracy_on_series(index):
    # combo = [name for name, ind in zip(datasets, index) if ind]
    combo = [name for name in datasets if name[0].upper() in index]
    return get_accuracy(combo)

# combos = [c for i in range(len(datasets)) for c in combinations(datasets, i)]
# index = pd.MultiIndex.from_product([[False, True]] * 4, names=datasets)
# series = index.to_series()#pd.Series(index=index, dtype=float)
# series = series[series != (False,)*4]
# series = series.map(get_accuracy_on_series)

xs = [
    "Lxxx", "xExx", "xxMx", "xxxS", 
    "LExx", "xEMx", "LxMx", "LxxS",
    "xExS", "xxMS", "LEMx", "LExS",
    "LxMS", "xEMS", "LEMS"
]
ys = [get_accuracy_on_series(x) for x in xs]
xs.append("xxxx")
ys.append([0.283, 0.2])
xs = ["\n".join([l for l in label]) for label in xs]
xs = [x.replace("x", "*") for x in xs]
firsts = [y[0] for y in ys]
seconds = [y[1] for y in ys]
means = [(y[0]+y[1])/2 for y in ys]

# Zip the lists together
combined = list(zip(xs, firsts, seconds, means))

# Sort by the third element (means)
combined_sorted = sorted(combined, key=lambda x: x[-1])

# Unzip them back into separate lists
xs_sorted, firsts_sorted, seconds_sorted, means_sorted = zip(*combined_sorted)

# Convert the tuples back into lists
xs = list(xs_sorted)
firsts = list(firsts_sorted)
seconds = list(seconds_sorted)
means = list(means_sorted)

plt.scatter(xs, firsts, alpha=0.3, color="r")
plt.scatter(xs, seconds, alpha=0.3, color="r")
plt.scatter(xs, means, alpha=1, color="r", marker="x")

# UpSet(series, sort_by="cardinality", with_lines=False, show_counts="%.2f").plot()
# axs = upset.plot()
plt.ylabel("Eval accuracy")
plt.xlabel("Train datasets")
# del axs["totals"]
plt.tight_layout()
plt.savefig("main_figure.png")