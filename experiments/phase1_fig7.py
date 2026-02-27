"""Phase 1 — Fig 7 reproduction: Example 2 (polynomial), λ=0.

Paper settings: D=40, m=100, M=10000, l=64, lr=0.01, C=1, λ=0, 10000 epochs
Methods: PM-Euler (SGD), Vanilla SAV, Restart SAV, Relax SAV

Usage:
    python experiments/phase1_fig7.py              # full 10k epochs
    python experiments/phase1_fig7.py --smoke      # smoke test (100 epochs)
"""

import argparse
import os
import sys
import time
import torch
import wandb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import generate_example2
from src.utils.trainer import get_device
from src.models.network import OneHiddenLayerNet
from src.algorithms.baselines import train_sgd
from src.algorithms.sav import train_vanilla_sav, train_restart_sav, train_relax_sav
from src.utils.plotting import plot_loss_curves, plot_energy_curves, plot_r_values
from src.utils.slack import send_slack

WANDB_PROJECT = "sav-gradient-flow-research"

# Paper's Fig 7 settings
D = 40
M_TOTAL = 10000
M_NEURONS = 100
BATCH_SIZE = 64
LR = 0.01  # dt for SAV, lr for SGD
C = 1.0
LAMBDA = 0.0
FULL_EPOCHS = 10000


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

    # Data — paper's settings
    X_tr, y_tr, X_te, y_te = generate_example2(D=D, M=M_TOTAL, seed=42)
    print(f"Data: X_train {X_tr.shape}, X_test {X_te.shape}")
    print(f"X_train mean={X_tr.mean():.4f}, std={X_tr.std():.4f}")

    out_dir = "results/phase1_sav"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te},
               os.path.join(out_dir, "fig7_data.pt"))

    base_config = {
        "figure": "fig7", "example": "example2_poly",
        "D": D, "m": M_NEURONS, "M": M_TOTAL,
        "batch_size": BATCH_SIZE, "lr": LR, "C": C, "lambda": LAMBDA,
        "epochs": epochs, "seed": 42,
    }

    all_results = []
    t_total = time.time()

    methods = [
        ("SGD", {"lr": LR, "momentum": 0},
         lambda net, wb: train_sgd(
            net, X_tr, y_tr, X_te, y_te,
            lr=LR, batch_size=BATCH_SIZE, epochs=epochs, device=device,
            slack_interval=slack_interval, wandb_run=wb)),
        ("Vanilla_SAV", {"C": C, "lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_vanilla_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=C, lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Restart_SAV", {"C": C, "lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_restart_sav(
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
            name=f"fig7-{name}-{tag}",
            config=run_config,
            tags=["phase1", "fig7", "example2", tag, name],
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
        torch.save(result, os.path.join(out_dir, f"fig7_{fname}.pt"))
        print(f"  Final train_loss={result['final_train_loss']:.4e}, "
              f"test_loss={result['final_test_loss']:.4e}, "
              f"wall_time={result['wall_time']:.1f}s")

    # Save combined results
    torch.save(all_results, os.path.join(out_dir, "fig7_all_results.pt"))

    # Plots
    plot_loss_curves(all_results, "Fig 7: Example 2 (D=40, m=100, lr=0.01, λ=0)",
                     os.path.join(out_dir, "fig7_loss_curves.png"))
    sav_results = [r for r in all_results if "SAV" in r["method"]]
    plot_energy_curves(sav_results, "Fig 7: Example 2",
                       os.path.join(out_dir, "fig7_energy_curves.png"))
    plot_r_values(sav_results, "Fig 7: Example 2",
                  os.path.join(out_dir, "fig7_r_values.png"))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Fig 7 COMPLETE | Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")

    print(f"\n{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['method']:<20} {r['final_train_loss']:>12.4e} "
              f"{r['final_test_loss']:>12.4e} {r['wall_time']:>7.1f}s")

    send_slack(
        f"Phase 1 Fig 7 COMPLETE | {epochs} epochs | "
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
