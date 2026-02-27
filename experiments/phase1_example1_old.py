"""Phase 1 — Example 1 (sin+cos): Run all 5 methods and save results.

Usage:
    python experiments/phase1_example1.py              # full 50k epochs
    python experiments/phase1_example1.py --smoke      # smoke test (100 epochs)
"""

import argparse
import os
import sys
import time
import torch
import wandb

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import generate_example1
from src.utils.trainer import get_device
from src.models.network import OneHiddenLayerNet
from src.algorithms.baselines import train_sgd, train_adam
from src.algorithms.sav import train_vanilla_sav, train_restart_sav, train_relax_sav
from src.utils.plotting import plot_loss_curves, plot_energy_curves, plot_r_values
from src.utils.slack import send_slack

WANDB_PROJECT = "sav-gradient-flow-research"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test (100 epochs)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs or (100 if args.smoke else 50000)
    slack_interval = 50 if args.smoke else 5000
    tag = "smoke" if args.smoke else "full"

    device = get_device()
    print(f"Device: {device}")

    # Data
    X_tr, y_tr, X_te, y_te = generate_example1(D=20, N_train=1000, N_test=200, seed=42)

    # Save data
    out_dir = "results/phase1_sav"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te},
               os.path.join(out_dir, "data_example1.pt"))

    # Common config for wandb
    base_config = {
        "example": "example1_sincos",
        "D": 20, "m": 100, "N_train": 1000, "N_test": 200,
        "batch_size": 256, "epochs": epochs, "seed": 42,
    }

    all_results = []
    t_total = time.time()

    methods = [
        ("SGD", {"lr": 0.1, "momentum": 0},
         lambda net, wb: train_sgd(
            net, X_tr, y_tr, X_te, y_te,
            lr=0.1, batch_size=256, epochs=epochs, device=device,
            slack_interval=slack_interval, wandb_run=wb)),
        ("Adam", {"lr": 0.001, "betas": (0.9, 0.999)},
         lambda net, wb: train_adam(
            net, X_tr, y_tr, X_te, y_te,
            lr=0.001, batch_size=256, epochs=epochs, device=device,
            slack_interval=slack_interval, wandb_run=wb)),
        ("Vanilla_SAV", {"C": 1.0, "lambda": 0.0, "dt": 0.1},
         lambda net, wb: train_vanilla_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Restart_SAV", {"C": 1.0, "lambda": 0.0, "dt": 0.1},
         lambda net, wb: train_restart_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Relax_SAV", {"C": 1.0, "lambda": 0.0, "dt": 0.1},
         lambda net, wb: train_relax_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
    ]

    for name, method_config, train_fn in methods:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        # Init wandb run
        run_config = {**base_config, **method_config, "method": name}
        wb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"phase1-ex1-{name}-{tag}",
            config=run_config,
            tags=["phase1", "example1", tag, name],
            reinit=True,
        )

        net = OneHiddenLayerNet(D=20, m=100, seed=42).to(device)
        result = train_fn(net, wb_run)
        all_results.append(result)

        wb_run.summary["final_train_loss"] = result["final_train_loss"]
        wb_run.summary["final_test_loss"] = result["final_test_loss"]
        wb_run.summary["wall_time"] = result["wall_time"]
        wb_run.finish()

        # Save per-method result
        fname = name.lower()
        torch.save(result, os.path.join(out_dir, f"example1_{fname}.pt"))
        print(f"  Final train_loss={result['final_train_loss']:.4e}, "
              f"test_loss={result['final_test_loss']:.4e}, "
              f"wall_time={result['wall_time']:.1f}s")

    # Save combined results
    torch.save(all_results, os.path.join(out_dir, "example1_all_results.pt"))

    # Plots
    plot_loss_curves(all_results, "Example 1 (sin+cos)",
                     os.path.join(out_dir, "example1_loss_curves.png"))
    sav_results = [r for r in all_results if "SAV" in r["method"]]
    plot_energy_curves(sav_results, "Example 1 (sin+cos)",
                       os.path.join(out_dir, "example1_energy_curves.png"))
    plot_r_values(sav_results, "Example 1 (sin+cos)",
                  os.path.join(out_dir, "example1_r_values.png"))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Example 1 COMPLETE | Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['method']:<20} {r['final_train_loss']:>12.4e} "
              f"{r['final_test_loss']:>12.4e} {r['wall_time']:>7.1f}s")

    send_slack(
        f"Phase 1 Example 1 COMPLETE | {epochs} epochs | "
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
