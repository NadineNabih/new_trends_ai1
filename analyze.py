"""Summarise representations_*_s*.csv: mean +- std over seeds, with chance and noise controls.
usage: python analyze.py results"""
import sys, glob
import pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "results"
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{out}/representations_*_s*.csv")])
cols = ["top5", "mrr", "avg_top5", "avg_mrr", "top5_perm_p", "chance_top5", "avg_chance_top5"]
cols = [c for c in cols if c in d]
g = d.groupby(["dataset", "split", "control", "recipe"])[cols]
t = g.mean().round(3).join(g.std().round(3)["top5"].rename("top5_std")).join(g.size().rename("n_seeds"))
pd.set_option("display.width", 250, "display.max_rows", 500)
for split in ["test", "test_seen_sent", "test_seen_subj"]:
    print(f"\n=== {split}")
    print(t.xs(split, level="split").to_string())
