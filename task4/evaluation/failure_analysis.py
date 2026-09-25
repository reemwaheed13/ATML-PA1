"""
Per-example acceptance/rejection analysis for Task 4 OSR.
Recomputes MLS score from cached logits (score itself isn't cached,
only logits — score = -max_k(logits)).
"""
import argparse
import torch
import numpy as np
import pandas as pd

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                    "dog", "frog", "horse", "ship", "truck"]

NEAR_CLASSES = {"bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel"}
FAR_CLASSES = {"bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe"}

TAU = -5.725428104400635  # from task4_final.json -> vanilla_mls_tau


def load_cache(model_name, cache_dir="task4/cache"):
    return torch.load(f"{cache_dir}/{model_name}.pt", map_location="cpu", weights_only=False)


def _to_numpy(x):
    return x.numpy() if torch.is_tensor(x) else np.array(x)


def analyze(cache, tau=TAU):
    splits = cache["splits"]

    logits_list = []
    true_class_list = []

    # known classes: CIFAR-10 test split, labels are integer indices
    test = splits["test"]
    test_logits = _to_numpy(test["logits"])
    test_labels = _to_numpy(test["labels"])
    logits_list.append(test_logits)
    true_class_list.extend(CIFAR10_CLASSES[l] for l in test_labels)

    # unknown classes: near/far splits already carry string class_names
    for grp in ["near", "far"]:
        s = splits[grp]
        logits_list.append(_to_numpy(s["logits"]))
        true_class_list.extend(list(s["class_names"]))

    logits_np = np.concatenate(logits_list, axis=0)
    scores = -logits_np.max(axis=1)
    preds = logits_np.argmax(axis=1)
    pred_labels = [CIFAR10_CLASSES[p] for p in preds]
    accepted = scores <= tau

    df = pd.DataFrame({
        "true_class": true_class_list,
        "pred_label": pred_labels,
        "score": scores,
        "accepted": accepted,
    })
    df["group"] = df["true_class"].apply(
        lambda c: "near" if c in NEAR_CLASSES else ("far" if c in FAR_CLASSES else "other")
    )
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["vanilla", "gcsc", "proser"])
    parser.add_argument("--cache_dir", default="task4/cache")
    parser.add_argument("--tau", type=float, default=TAU)
    args = parser.parse_args()

    cache = load_cache(args.model, args.cache_dir)
    df = analyze(cache, tau=args.tau)

    print("Acceptance rate by group and class:")
    print(df.groupby(["group", "true_class"])["accepted"].mean().sort_values(ascending=False))

    for grp in ["near", "far"]:
        sub = df[(df.accepted) & (df.group == grp)]
        print(f"\nCrosstab — {grp} unknowns:")
        print(pd.crosstab(sub["true_class"], sub["pred_label"]))

    out_path = f"results/task4/failure_analysis_{args.model}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved full per-example results to {out_path}")


if __name__ == "__main__":
    main()
