"""Phase 3 — IEQ on Example 2 (polynomial), λ=0.

Same data settings as Phase 1/2: D=40, m=100, M=10000, batch_size=64, lr=0.01, λ=0
Methods: SGD (baseline), Vanilla IEQ, Restart IEQ, Relax IEQ
Note: IEQ requires Jacobian computation — much slower per step. Run 5000 epochs.

Usage:
    python experiments/phase3_example2.py              # full 5k epochs
    python experiments/phase3_example2.py --smoke      # smoke test (50 epochs)
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
from src.algorithms.ieq import train_vanilla_ieq, train_restart_ieq, train_relax_ieq
from src.utils.plotting import plot_loss_curves, plot_energy_curves, plot_r_values
from src.utils.slack import send_slack

WANDB_PROJECT = "sav-gradient-flow-research"

D = 40
M_TOTAL = 10000
M_NEURONS = 100
BATCH_SIZE = 64
LR = 0.01
LAMBDA = 0.0
FULL_EPOCHS = 5000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test (50 epochs)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs or (50 if args.smoke else FULL_EPOCHS)
    slack_interval = 25 if args.smoke else 500
    tag = "smoke" if args.smoke else "full"

    device = get_device()
    print(f"Device: {device}")

    X_tr, y_tr, X_te, y_te = generate_example2(D=D, M=M_TOTAL, seed=42)
    print(f"Data: X_train {X_tr.shape}, X_test {X_te.shape}")

    out_dir = "results/phase3_ieq"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te},
               os.path.join(out_dir, "example2_data.pt"))

    base_config = {
        "phase": "phase3", "example": "example2_poly",
        "D": D, "m": M_NEURONS, "M": M_TOTAL,
        "batch_size": BATCH_SIZE, "lr": LR, "lambda": LAMBDA,
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
        ("Vanilla_IEQ", {"lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_vanilla_ieq(
            net, X_tr, y_tr, X_te, y_te,
            lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Restart_IEQ", {"lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_restart_ieq(
            net, X_tr, y_tr, X_te, y_te,
            lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
        ("Relax_IEQ", {"lambda": LAMBDA, "dt": LR},
         lambda net, wb: train_relax_ieq(
            net, X_tr, y_tr, X_te, y_te,
            lambda_=LAMBDA, dt=LR, batch_size=BATCH_SIZE, epochs=epochs,
            device=device, slack_interval=slack_interval, wandb_run=wb)),
    ]

    for name, method_config, train_fn in methods:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        run_config = {**base_config, **method_config, "method": name}
        wb_run = wandb.init(
            project=WANDB_PROJECT,
            name=f"p3-ex2-{name}-{tag}",
            config=run_config,
            tags=["phase3", "example2", tag, name],
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
        torch.save(result, os.path.join(out_dir, f"example2_{fname}.pt"))
        print(f"  Final train_loss={result['final_train_loss']:.4e}, "
              f"test_loss={result['final_test_loss']:.4e}, "
              f"wall_time={result['wall_time']:.1f}s")

    torch.save(all_results, os.path.join(out_dir, "example2_all_results.pt"))

    plot_loss_curves(all_results,
                     "Phase 3 IEQ: Example 2 (D=40, m=100, lr=0.01, \u03bb=0)",
                     os.path.join(out_dir, "example2_loss_curves.png"))
    ieq_results = [r for r in all_results if "IEQ" in r["method"]]
    plot_energy_curves(ieq_results, "Phase 3 IEQ: Example 2",
                       os.path.join(out_dir, "example2_energy_curves.png"))
    plot_r_values(ieq_results, "Phase 3 IEQ: Example 2 (||q|| trajectory)",
                  os.path.join(out_dir, "example2_r_values.png"))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Phase 3 Example 2 COMPLETE | Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")

    print(f"\n{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['method']:<20} {r['final_train_loss']:>12.4e} "
              f"{r['final_test_loss']:>12.4e} {r['wall_time']:>7.1f}s")

    send_slack(
        f"Phase 3 Example 2 COMPLETE | {epochs} epochs | "
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
