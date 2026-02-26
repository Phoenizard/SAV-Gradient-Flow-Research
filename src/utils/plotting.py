"""Plotting utilities for loss and energy curves."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curves(results_list, title, save_path):
    """Plot train and test loss curves for multiple methods.

    Args:
        results_list: list of result dicts with 'method', 'train_loss', 'test_loss'
        title: plot title
        save_path: path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for res in results_list:
        epochs = range(1, len(res["train_loss"]) + 1)
        ax1.plot(epochs, res["train_loss"], label=res["method"])
        ax2.plot(epochs, res["test_loss"], label=res["method"])

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"{title} — Train Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_yscale("log")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Loss")
    ax2.set_title(f"{title} — Test Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved loss curves to {save_path}")


def plot_energy_curves(results_list, title, save_path):
    """Plot energy curves for SAV methods.

    Args:
        results_list: list of result dicts with 'method', 'energy'
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results_list:
        if not res["energy"]:
            continue
        epochs = range(1, len(res["energy"]) + 1)
        ax.plot(epochs, res["energy"], label=res["method"])

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy")
    ax.set_title(f"{title} — Energy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved energy curves to {save_path}")


def plot_r_values(results_list, title, save_path):
    """Plot r auxiliary variable evolution for SAV methods.

    Args:
        results_list: list of result dicts with 'method', 'r_values'
        title: plot title
        save_path: path to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for res in results_list:
        if not res.get("r_values"):
            continue
        epochs = range(1, len(res["r_values"]) + 1)
        ax.plot(epochs, res["r_values"], label=res["method"])

    ax.set_xlabel("Epoch")
    ax.set_ylabel("r (auxiliary variable)")
    ax.set_title(f"{title} — Auxiliary Variable r")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved r-values plot to {save_path}")
