"""Phase 1 — Fig 1 reproduction: Energy verification, Example 1, λ=10.

Paper settings: D=20, m=1000, M=10000, l=8000(full batch), lr=0.6, C=1, λ=10, 8000 epochs
Methods: GD (SGD full-batch), Vanilla SAV, Relax SAV
Purpose: Verify energy is non-increasing (energy stability test)

Usage:
    python experiments/phase1_fig1.py              # full 8k epochs
    python experiments/phase1_fig1.py --smoke      # smoke test (100 epochs)
"""

import argparse
import os
import sys
import time
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import generate_example1
from src.utils.trainer import get_device
from src.models.network import OneHiddenLayerNet
from src.algorithms.baselines import train_sgd
from src.algorithms.sav import train_vanilla_sav, train_relax_sav
from src.utils.plotting import plot_loss_curves, plot_energy_curves, plot_r_values
from src.utils.slack import send_slack

WANDB_PROJECT = "sav-gradient-flow-research"

# Paper's Fig 1 settings
D = 20
M_TOTAL = 10000
M_NEURONS = 1000
BATCH_SIZE = 8000  # Full batch (80% of 10000)
LR = 0.6  # dt for SAV, lr for SGD
C = 1.0
LAMBDA = 10.0
FULL_EPOCHS = 8000


def check_energy_stability(energy_values, method_name):
    """Check if energy is non-increasing. Report violations."""
    violations = 0
    max_increase = 0.0
    for i in range(1, len(energy_values)):
        increase = energy_values[i] - energy_values[i-1]
        if increase > 1e-10:  # small tolerance for floating point
            violations += 1
            max_increase = max(max_increase, increase)
    if violations == 0:
        print(f"  {method_name}: Energy STABLE (non-increasing for all {len(energy_values)} epochs)")
    else:
        print(f"  {method_name}: Energy UNSTABLE! {violations} violations, max increase={max_increase:.4e}")
    return violations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test (100 epochs)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs or (100 if args.smoke else FULL_EPOCHS)
    slack_interval = 50 if args.smoke else 1000
    tag = "smoke" if args.smoke else "full"

    device = get_device()
    print(f"Device: {device}")

    # Data — paper's settings (D=20 for Fig 1)
    X_tr, y_tr, X_te, y_te = generate_example1(D=D, M=M_TOTAL, seed=42)
    print(f"Data: X_train {X_tr.shape}, X_test {X_te.shape}")
    print(f"Effective batch size: {BATCH_SIZE} (full batch = {len(X_tr)})")

    out_dir = "results/phase1_sav"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te},
               os.path.join(out_dir, "fig1_data.pt"))

    base_config = {
        "figure": "fig1", "example": "example1_sincos",
        "D": D, "m": M_NEURONS, "M": M_TOTAL,
        "batch_size": BATCH_SIZE, "lr": LR, "C": C, "lambda": LAMBDA,
        "epochs": epochs, "seed": 42,
    }

    all_results = []
    t_total = time.time()

    methods = [
        ("GD", {"lr": LR, "momentum": 0},
         lambda net, wb: train_sgd(
            net, X_tr, y_tr, X_te, y_te,
            lr=LR, batch_size=BATCH_SIZE, epochs=epochs, device=device,
            slack_interval=slack_interval, wandb_run=wb)),
        ("Vanilla_SAV", {"C": C, "lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_vanilla_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=C, lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Relax_SAV", {"C": C, "lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_relax_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=C, lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
    ]

    for name, method_config, train_fn in methods:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        run_config = {**base_config, **method_config, "method": name}
        wb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"fig1-{name}-{tag}",
            config=run_config,
            tags=["phase1", "fig1", "energy", tag, name],
            reinit=True,
        )

        net = OneHiddenLayerNet(D=D, m=M_NEURONS, seed=42).to(device)
        result = train_fn(net, wb_run)
        all_results.append(result)

        wb_run.summary["final_train_loss"] = result["final_train_loss"]
        wb_run.summary["final_test_loss"] = result["final_test_loss"]
        wb_run.summary["wall_time"] = result["wall_time"]
        wb_run.finish()

        fname = name.lower()
        torch.save(result, os.path.join(out_dir, f"fig1_{fname}.pt"))
        print(f"  Final train_loss={result['final_train_loss']:.4e}, "
              f"test_loss={result['final_test_loss']:.4e}, "
              f"wall_time={result['wall_time']:.1f}s")

    # Save combined results
    torch.save(all_results, os.path.join(out_dir, "fig1_all_results.pt"))

    # Plots
    plot_loss_curves(all_results, "Fig 1: Example 1 (D=20, m=1000, lr=0.6, λ=10)",
                     os.path.join(out_dir, "fig1_loss_curves.png"))
    sav_results = [r for r in all_results if "SAV" in r["method"]]
    plot_energy_curves(sav_results, "Fig 1: Energy Stability (λ=10)",
                       os.path.join(out_dir, "fig1_energy_curves.png"))
    plot_r_values(sav_results, "Fig 1: Example 1",
                  os.path.join(out_dir, "fig1_r_values.png"))

    # Energy stability check
    print(f"\n{'='*60}")
    print("  ENERGY STABILITY CHECK")
    print(f"{'='*60}")
    for r in all_results:
        if r["energy"]:
            check_energy_stability(r["energy"], r["method"])

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Fig 1 COMPLETE | Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")

    print(f"\n{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['method']:<20} {r['final_train_loss']:>12.4e} "
              f"{r['final_test_loss']:>12.4e} {r['wall_time']:>7.1f}s")

    send_slack(
        f"Phase 1 Fig 1 (Energy) COMPLETE | {epochs} epochs | "
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
