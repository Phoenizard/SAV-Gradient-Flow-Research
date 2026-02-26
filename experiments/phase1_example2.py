"""Phase 1 — Example 2 (polynomial): Run all 5 methods and save results.

Usage:
    python experiments/phase1_example2.py              # full 50k epochs
    python experiments/phase1_example2.py --smoke      # smoke test (100 epochs)
"""

import argparse
import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data import generate_example2
from src.utils.trainer import get_device
from src.models.network import OneHiddenLayerNet
from src.algorithms.baselines import train_sgd, train_adam
from src.algorithms.sav import train_vanilla_sav, train_restart_sav, train_relax_sav
from src.utils.plotting import plot_loss_curves, plot_energy_curves, plot_r_values
from src.utils.slack import send_slack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Smoke test (100 epochs)")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    epochs = args.epochs or (100 if args.smoke else 50000)
    slack_interval = 50 if args.smoke else 1000

    device = get_device()
    print(f"Device: {device}")

    # Data
    X_tr, y_tr, X_te, y_te = generate_example2(D=20, N_train=1000, N_test=200, seed=42)

    # Save data
    out_dir = "results/phase1_sav"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te},
               os.path.join(out_dir, "data_example2.pt"))

    all_results = []
    t_total = time.time()

    methods = [
        ("SGD", lambda net: train_sgd(
            net, X_tr, y_tr, X_te, y_te,
            lr=0.1, batch_size=256, epochs=epochs, device=device,
            slack_interval=slack_interval)),
        ("Adam", lambda net: train_adam(
            net, X_tr, y_tr, X_te, y_te,
            lr=0.001, batch_size=256, epochs=epochs, device=device,
            slack_interval=slack_interval)),
        ("Vanilla SAV", lambda net: train_vanilla_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval)),
        ("Restart SAV", lambda net: train_restart_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval)),
        ("Relax SAV", lambda net: train_relax_sav(
            net, X_tr, y_tr, X_te, y_te,
            C=1.0, lambda_=0.0, dt=0.1, batch_size=256, epochs=epochs,
            device=device, slack_interval=slack_interval)),
    ]

    for name, train_fn in methods:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")

        net = OneHiddenLayerNet(D=20, m=100, seed=42).to(device)
        result = train_fn(net)
        all_results.append(result)

        fname = name.lower().replace(" ", "_")
        torch.save(result, os.path.join(out_dir, f"example2_{fname}.pt"))
        print(f"  Final train_loss={result['final_train_loss']:.4e}, "
              f"test_loss={result['final_test_loss']:.4e}, "
              f"wall_time={result['wall_time']:.1f}s")

    # Save combined results
    torch.save(all_results, os.path.join(out_dir, "example2_all_results.pt"))

    # Plots
    plot_loss_curves(all_results, "Example 2 (polynomial)",
                     os.path.join(out_dir, "example2_loss_curves.png"))
    sav_results = [r for r in all_results if "SAV" in r["method"]]
    plot_energy_curves(sav_results, "Example 2 (polynomial)",
                       os.path.join(out_dir, "example2_energy_curves.png"))
    plot_r_values(sav_results, "Example 2 (polynomial)",
                  os.path.join(out_dir, "example2_r_values.png"))

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"Example 2 COMPLETE | Total wall time: {total_time:.1f}s")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Method':<20} {'Train Loss':>12} {'Test Loss':>12} {'Time':>8}")
    print("-" * 56)
    for r in all_results:
        print(f"{r['method']:<20} {r['final_train_loss']:>12.4e} "
              f"{r['final_test_loss']:>12.4e} {r['wall_time']:>7.1f}s")

    send_slack(
        f"Phase 1 Example 2 COMPLETE | {epochs} epochs | "
        f"Total time: {total_time:.1f}s"
    )


if __name__ == "__main__":
    main()
