"""CLI.   python run.py prepare|inspect|run|plots ...   (see README.md)"""
import argparse
from dataclasses import fields

from zuco_eeg.data import build_cache, inspect_mat
from zuco_eeg.experiments import exp_cross, exp_representations, exp_robustness, make_plots
from zuco_eeg.tasks import exp_classification, exp_cls_robust
from zuco_eeg.train import Config

p = argparse.ArgumentParser()
sub = p.add_subparsers(dest="cmd", required=True)

a = sub.add_parser("inspect", help="print the structure of one results*.mat file")
a.add_argument("mat")

a = sub.add_parser("prepare", help="preprocess + cache a ZuCo dataset")
a.add_argument("--src", required=True, help="folder containing results*.mat (searched recursively)")
a.add_argument("--dataset", required=True, choices=["zuco1", "zuco2"])
a.add_argument("--cache_root", default="cache")
a.add_argument("--fs", type=int, default=128)
a.add_argument("--max_sec", type=float, default=12.0)
a.add_argument("--max_files", type=int, default=None)

a = sub.add_parser("run", help="run experiments")
a.add_argument("--exp", default="all", choices=["representations", "robustness", "cross", "all", "sentiment", "subject", "subject_robust"])
for f in fields(Config):
    if f.type in (int, float, str):
        a.add_argument(f"--{f.name}", type=f.type, default=None)
a.add_argument("--datasets", nargs="+", default=None)
a.add_argument("--seeds", nargs="+", type=int, default=None, help="e.g. 0 1 2 (different subject/sentence splits)")
a.add_argument("--tasks", nargs="+", default=None, help="e.g. NR SR TSR (default: all)")

sub.add_parser("plots").add_argument("--out_dir", default="results")
args = p.parse_args()

if args.cmd == "inspect":
    inspect_mat(args.mat)
elif args.cmd == "prepare":
    build_cache(args.src, args.cache_root, args.dataset, args.fs, args.max_sec, max_files=args.max_files)
elif args.cmd == "plots":
    make_plots(Config(out_dir=args.out_dir))
else:
    cfg = Config()
    for f in fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    if args.datasets:
        cfg.datasets = tuple(args.datasets)
    if args.tasks:
        cfg.tasks = tuple(args.tasks)
    for seed in (args.seeds or [cfg.seed]):
        cfg.seed = seed
        print(f"\n########## seed {seed}")
        if args.exp in ("representations", "all"):
            for ds in cfg.datasets:
                exp_representations(cfg, ds)
        if args.exp in ("sentiment", "subject"):
            for ds in cfg.datasets:
                exp_classification(cfg, ds, args.exp)
        if args.exp == "subject_robust":
            for ds in cfg.datasets:
                exp_cls_robust(cfg, ds, "subject")
        if args.exp in ("robustness", "all"):
            for ds in cfg.datasets:
                exp_robustness(cfg, ds)
        if args.exp in ("cross", "all") and len(cfg.datasets) == 2:
            d1, d2 = cfg.datasets
            exp_cross(cfg, d1, d2)
            exp_cross(cfg, d2, d1)
    make_plots(cfg)
