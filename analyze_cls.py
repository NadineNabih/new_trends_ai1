"""python analyze_cls.py results  -> mean over seeds for cls_*.csv and clsrob_*.csv"""
import sys, glob
import pandas as pd
out = sys.argv[1] if len(sys.argv) > 1 else "results"
pd.set_option("display.width", 200, "display.max_rows", 500)
f = glob.glob(f"{out}/cls_*_s*.csv")
if f:
    d = pd.concat(pd.read_csv(x) for x in f)
    cols = ["bal_acc", "chance_bal_acc", "avg_bal_acc", "macro_f1", "bal_acc_perm_p"]
    g = d.groupby(["task", "dataset", "split", "control", "recipe"])[cols]
    print(g.mean().round(3).join(g.std().round(3)["bal_acc"].rename("std")).join(g.size().rename("n_seeds")).to_string())
f = glob.glob(f"{out}/clsrob_*_s*.csv")
if f:
    d = pd.concat(pd.read_csv(x) for x in f)
    print("\nrobustness: balanced accuracy (mean over seeds)")
    print(d.pivot_table(index=["task", "kind", "level"], columns="variant", values="bal_acc").round(3).to_string())
