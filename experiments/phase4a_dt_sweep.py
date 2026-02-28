"""Phase 4A — dt sensitivity sweep for Example 1 and Example 2.

Sweep dt = {0.01, 0.05, 0.1, 0.2, 0.5, 1.0} for:
  - SGD (baseline)
  - Restart SAV
  - Restart ESAV
  - Restart IEQ (only dt=0.1, 0.2 — too slow for full sweep)

Goal: Show SAV/ESAV remain stable at large dt where SGD diverges.
This is a key figure for the SISC paper.

Usage:
    python experiments/phase4a_dt_sweep.py --example 1           # Example 1
    python experiments/phase4a_dt_sweep.py --example 2           # Example 2
    python experiments/phase4a_dt_sweep.py --example 1 --smoke   # smoke test
"""

import argparse
import os
import sys
import time
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import generate_example1, generate_example2
from src.utils.trainer import get_device
from src.models.network import OneHiddenLayerNet
from src.algorithms.baselines import train_sgd
from src.algorithms.sav import train_restart_sav
from src.algorithms.esav import train_restart_esav
from src.algorithms.ieq import train_restart_ieq
from src.utils.slack import send_slack

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WANDB_PROJECT = "sav-gradient-flow-research"

# Example-specific settings (matching Phase 1/2/3)
EXAMPLE_CONFIG = {
    1: dict(D=40, M=10000, m=1000, batch_size=64, base_dt=0.2, C=1, lambda_=0.0,
            gen_fn="generate_example1", label="sin+cos"),
    2: dict(D=40, M=10000, m=100, batch_size=64, base_dt=0.01, C=1, lambda_=0.0,
            gen_fn="generate_example2", label="polynomial"),
}

# dt values to sweep
DT_VALUES_EX1 = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
DT_VALUES_EX2 = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]


def plot_dt_sweep(all_results, title, save_path):
    """Plot final test loss vs dt for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods = sorted(set(r["method"] for r in all_results))
    colors = {"SGD": "red", "Restart SAV": "blue", "Restart ESAV": "green",
              "Restart IEQ": "purple"}
    markers = {"SGD": "o", "Restart SAV": "s", "Restart ESAV": "^",
               "Restart IEQ": "D"}

    for ax, metric, ylabel in [
        (axes[0], "final_train_loss", "Train Relative Error"),
        (axes[1], "final_test_loss", "Test Relative Error"),
    ]:
        for method in methods:
            pts = [(r["dt"], r[metric]) for r in all_results if r["method"] == method]
            pts.sort()
            dts, vals = zip(*pts)
            ax.semilogy(dts, vals, "-" + markers.get(method, "o"),
                        color=colors.get(method, "gray"),
                        label=method, markersize=6, linewidth=1.5)

        ax.set_xlabel("dt (learning rate)")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dt sweep plot to {save_path}")


def plot_dt_loss_curves(all_results, dt_values, title, save_path):
    """Plot loss curves for each dt value side by side."""
    n_dt = len(dt_values)
    fig, axes = plt.subplots(1, n_dt, figsize=(4 * n_dt, 4), sharey=True)
    if n_dt == 1:
        axes = [axes]

    colors = {"SGD": "red", "Restart SAV": "blue", "Restart ESAV": "green",
              "Restart IEQ": "purple"}

    for i, dt in enumerate(dt_values):
        ax = axes[i]
        dt_results = [r for r in all_results if abs(r["dt"] - dt) < 1e-8]
        for r in dt_results:
            losses = r["test_loss"]
            ax.semilogy(range(1, len(losses) + 1), losses,
                        color=colors.get(r["method"], "gray"),
                        label=r["method"], linewidth=1, alpha=0.8)
        ax.set_title(f"dt = {dt}")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.set_ylabel("Test Relative Error")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dt loss curves to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--example", type=int, required=True, choices=[1, 2])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-ieq", action="store_true", help="Skip IEQ (slow)")
    args = parser.parse_args()

    cfg = EXAMPLE_CONFIG[args.example]
    dt_values = DT_VALUES_EX1 if args.example == 1 else DT_VALUES_EX2
    epochs = args.epochs or (50 if args.smoke else 5000)
    slack_interval = 25 if args.smoke else 1000
    tag = "smoke" if args.smoke else "full"

    device = get_device()
    print(f"Device: {device}")

    # Generate data
    gen_fn = generate_example1 if args.example == 1 else generate_example2
    X_tr, y_tr, X_te, y_te = gen_fn(D=cfg["D"], M=cfg["M"], seed=42)
    print(f"Example {args.example} ({cfg['label']}): X_train {X_tr.shape}")

    out_dir = f"results/phase4_dt_sweep"
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    t_total = time.time()

    # Methods to sweep (IEQ only at select dt values due to cost)
    ieq_dt_values = [cfg["base_dt"]] if not args.no_ieq else []

    for dt in dt_values:
        print(f"\n{'='*60}")
        print(f"  dt = {dt}")
        print(f"{'='*60}")

        methods = [
            ("SGD", lambda net, wb, dt=dt: train_sgd(
                net, X_tr, y_tr, X_te, y_te,
                lr=dt, batch_size=cfg["batch_size"], epochs=epochs,
                device=device, slack_interval=slack_interval, wandb_run=wb)),
            ("Restart SAV", lambda net, wb, dt=dt: train_restart_sav(
                net, X_tr, y_tr, X_te, y_te,
                C=cfg["C"], lambda_=cfg["lambda_"], dt=dt,
                batch_size=cfg["batch_size"], epochs=epochs,
                device=device, slack_interval=slack_interval, wandb_run=wb)),
            ("Restart ESAV", lambda net, wb, dt=dt: train_restart_esav(
                net, X_tr, y_tr, X_te, y_te,
                lambda_=cfg["lambda_"], dt=dt,
                batch_size=cfg["batch_size"], epochs=epochs,
                device=device, slack_interval=slack_interval, wandb_run=wb)),
        ]

        # Add IEQ only at base dt
        if dt in ieq_dt_values:
            methods.append(
                ("Restart IEQ", lambda net, wb, dt=dt: train_restart_ieq(
                    net, X_tr, y_tr, X_te, y_te,
                    lambda_=cfg["lambda_"], dt=dt,
                    batch_size=cfg["batch_size"], epochs=epochs,
                    device=device, slack_interval=slack_interval, wandb_run=wb))
            )

        for method_name, train_fn in methods:
            print(f"\n  --- {method_name} (dt={dt}) ---")

            run_name = f"p4a-ex{args.example}-{method_name.replace(' ','_')}-dt{dt}-{tag}"
            wb_run = wandb.init(
                project=WANDB_PROJECT, name=run_name,
                config={"phase": "phase4a", "example": args.example,
                        "method": method_name, "dt": dt, "epochs": epochs,
                        **cfg},
                tags=["phase4a", f"example{args.example}", tag, method_name],
                reinit=True,
            )

            net = OneHiddenLayerNet(D=cfg["D"], m=cfg["m"], seed=42).to(device)
            result = train_fn(net, wb_run)
            result["dt"] = dt  # tag with dt value

            wb_run.summary["final_train_loss"] = result["final_train_loss"]
            wb_run.summary["final_test_loss"] = result["final_test_loss"]
            wb_run.summary["wall_time"] = result["wall_time"]
            wb_run.summary["dt"] = dt
            wb_run.finish()

            all_results.append(result)
            print(f"    train={result['final_train_loss']:.4e} "
                  f"test={result['final_test_loss']:.4e} "
                  f"wall={result['wall_time']:.0f}s")

    # Save results
    torch.save(all_results,
               os.path.join(out_dir, f"example{args.example}_dt_sweep.pt"))

    # Plots
    plot_dt_sweep(
        all_results,
        f"Phase 4A: dt Sensitivity — Example {args.example} ({cfg['label']}), "
        f"{epochs} epochs",
        os.path.join(out_dir, f"example{args.example}_dt_sweep.png"))

    plot_dt_loss_curves(
        all_results, dt_values,
        f"Phase 4A: Loss Curves by dt — Example {args.example}",
        os.path.join(out_dir, f"example{args.example}_dt_loss_curves.png"))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Phase 4A Example {args.example} COMPLETE | {total_time:.0f}s")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Method':<18} {'dt':>6} {'Train':>12} {'Test':>12} {'Time':>8}")
    print("-" * 60)
    for r in sorted(all_results, key=lambda x: (x["method"], x["dt"])):
        print(f"{r['method']:<18} {r['dt']:>6.3f} "
              f"{r['final_train_loss']:>12.4e} {r['final_test_loss']:>12.4e} "
              f"{r['wall_time']:>7.0f}s")

    send_slack(
        f"Phase 4A dt sweep Example {args.example} COMPLETE | "
        f"{len(dt_values)} dt values x 3 methods | "
        f"{epochs} epochs | Total: {total_time:.0f}s"
    )


if __name__ == "__main__":
    main()
