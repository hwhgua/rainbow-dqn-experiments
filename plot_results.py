"""Create portfolio-ready figures from a controlled experiment directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# The CLI must work on Windows, CI, and other machines without a GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_results import write_aggregated_results


def _save_figure(figure, path, *, tight_layout_kwargs=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.tight_layout(**(tight_layout_kwargs or {}))
        figure.savefig(path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(figure)


def _load_training_logs(experiment_dir):
    """Load episode-level training histories from every completed run."""
    required_columns = {
        "run_id",
        "algorithm",
        "train_seed",
        "global_step_end",
        "score",
        "avg_score",
    }
    paths = sorted(Path(experiment_dir).rglob("training_log.csv"))
    if not paths:
        raise FileNotFoundError(f"找不到 training_log.csv：{experiment_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        missing = required_columns.difference(frame.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{path} 缺少訓練紀錄欄位：{missing_text}")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _complete_episodes(frame):
    """Exclude the synthetic partial episode created by a fixed step budget."""
    if "budget_cutoff" not in frame.columns:
        return frame
    budget_cutoff = frame["budget_cutoff"].astype(str).str.lower().eq("true")
    return frame.loc[~budget_cutoff]


def _trailing_episode_mean(returns):
    """Return the trailing mean over up to 100 complete episodes."""
    return pd.Series(returns).rolling(window=100, min_periods=1).mean().to_numpy()


def plot_training_convergence(experiment_dir, save_dir, algorithms, colors):
    """Plot exploratory episode returns and their recorded 100-episode mean.

    This is a training diagnostic. It intentionally remains separate from the
    deterministic evaluation curve because raw training returns include each
    algorithm's exploration behaviour.
    """
    training = _load_training_logs(experiment_dir)
    path = Path(save_dir) / "training_episode_return_convergence.png"

    columns = 2
    rows = int(np.ceil(len(algorithms) / columns))
    figure, axes = plt.subplots(
        rows, columns, figsize=(14, 3.8 * rows), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, algorithm in zip(axes, algorithms):
        algorithm_frame = training.loc[training["algorithm"] == algorithm]
        if algorithm_frame.empty:
            axis.set_visible(False)
            continue

        seeds = sorted(algorithm_frame["train_seed"].drop_duplicates())
        for seed in seeds:
            frame = algorithm_frame.loc[algorithm_frame["train_seed"] == seed].sort_values(
                "global_step_end"
            )
            frame = _complete_episodes(frame)
            if frame.empty:
                continue

            steps = frame["global_step_end"].to_numpy()
            returns = frame["score"].to_numpy()
            moving_average = _trailing_episode_mean(returns)
            seed_label = f"seed {seed}: " if len(seeds) > 1 else ""

            axis.scatter(
                steps,
                returns,
                color=colors[algorithm],
                alpha=0.16,
                s=10,
                linewidths=0,
                label=f"{seed_label}episode return",
            )
            axis.plot(
                steps,
                moving_average,
                color=colors[algorithm],
                linewidth=2.2,
                label=f"{seed_label}trailing mean (up to 100 complete episodes)",
            )

        axis.set_title(f"{algorithm} (training seeds: {len(seeds)})")
        axis.set_xlabel("Cumulative training environment steps")
        axis.set_ylabel("Exploratory episode return")
        axis.grid(True, linestyle="--", alpha=0.35)
        if len(seeds) > 1:
            axis.legend(fontsize=8, loc="lower right")

    for axis in axes[len(algorithms) :]:
        axis.set_visible(False)

    figure.suptitle("Training Convergence by DQN Variant", fontsize=16, y=0.995)
    figure.text(
        0.5,
        0.01,
        "Points are exploratory returns from complete episodes; lines are trailing means over up to 100 complete episodes. "
        "Use deterministic evaluation figures for fair model comparisons.",
        ha="center",
        fontsize=9,
    )
    _save_figure(figure, path, tight_layout_kwargs={"rect": (0, 0.04, 1, 0.96)})
    return path


def plot_training_convergence_overlay(experiment_dir, save_dir, algorithms, colors):
    """Overlay training trailing means without raw episode-return points."""
    training = _load_training_logs(experiment_dir)
    path = Path(save_dir) / "training_trailing_mean_overlay.png"
    figure, axis = plt.subplots(figsize=(12, 7.2))
    all_means = []
    all_steps = []

    for algorithm in algorithms:
        algorithm_frame = training.loc[training["algorithm"] == algorithm]
        if algorithm_frame.empty:
            continue

        seeds = sorted(algorithm_frame["train_seed"].drop_duplicates())
        for seed_index, seed in enumerate(seeds):
            frame = algorithm_frame.loc[algorithm_frame["train_seed"] == seed].sort_values(
                "global_step_end"
            )
            frame = _complete_episodes(frame)
            if frame.empty:
                continue

            steps = frame["global_step_end"].to_numpy()
            moving_average = _trailing_episode_mean(frame["score"].to_numpy())
            all_steps.extend(steps)
            all_means.extend(moving_average)
            label = f"{algorithm} (n=1)" if len(seeds) == 1 else f"{algorithm}, seed {seed}"
            line_styles = ("-", "--", ":", "-.")
            axis.plot(
                steps,
                moving_average,
                color=colors[algorithm],
                linewidth=2.35 if len(seeds) == 1 else 1.6,
                alpha=1.0 if len(seeds) == 1 else 0.72,
                linestyle=line_styles[seed_index % len(line_styles)],
                label=label,
            )

    if not all_steps:
        raise ValueError("找不到可繪製的完整訓練 episode")

    axis.set_xlim(0, max(all_steps))
    axis.set_ylim(bottom=0, top=max(all_means) * 1.06)
    axis.set_xlabel("Cumulative training environment steps")
    axis.set_ylabel("Exploratory episode return (trailing mean)")
    axis.set_title("Training Convergence: Trailing Mean of Complete Episodes")
    axis.grid(True, linestyle="--", alpha=0.35)
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(4, len(labels)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.06),
    )
    figure.text(
        0.5,
        0.01,
        "Each line is a trailing mean over up to 100 complete episodes from exploratory training returns. "
        "Use deterministic evaluation figures for fair policy-performance comparisons.",
        ha="center",
        fontsize=9,
    )
    _save_figure(figure, path, tight_layout_kwargs={"rect": (0, 0.13, 1, 1)})
    return path


def plot_training_convergence_only(experiment_dir, save_dir=None):
    """Create only the training-convergence diagnostic without overwriting evaluation figures."""
    experiment_dir = Path(experiment_dir)
    if save_dir is None:
        save_dir = experiment_dir / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training = _load_training_logs(experiment_dir)
    algorithms = list(training["algorithm"].drop_duplicates())
    palette = plt.get_cmap("tab10")
    colors = {algorithm: palette(index % 10) for index, algorithm in enumerate(algorithms)}
    return plot_training_convergence(experiment_dir, save_dir, algorithms, colors)


def plot_training_convergence_overlay_only(experiment_dir, save_dir=None):
    """Create only the overlaid training-convergence figure."""
    experiment_dir = Path(experiment_dir)
    if save_dir is None:
        save_dir = experiment_dir / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    training = _load_training_logs(experiment_dir)
    algorithms = list(training["algorithm"].drop_duplicates())
    palette = plt.get_cmap("tab10")
    colors = {algorithm: palette(index % 10) for index, algorithm in enumerate(algorithms)}
    return plot_training_convergence_overlay(experiment_dir, save_dir, algorithms, colors)


def plot_experiment(experiment_dir, save_dir=None):
    """Create deterministic evaluation figures and an exploratory training curve.

    The curve shading is one standard deviation across *training seed means*;
    it is deliberately not the variation between evaluation episodes.
    """
    experiment_dir = Path(experiment_dir)
    if save_dir is None:
        save_dir = experiment_dir / "figures"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    _, aggregate, auc_summary, aggregate_path, auc_path = write_aggregated_results(
        experiment_dir
    )
    algorithms = list(aggregate["algorithm"].drop_duplicates())
    palette = plt.get_cmap("tab10")
    colors = {algorithm: palette(index % 10) for index, algorithm in enumerate(algorithms)}

    curve_path = save_dir / "evaluation_return_vs_env_steps.png"
    figure, axis = plt.subplots(figsize=(11, 6.5))
    for algorithm in algorithms:
        frame = aggregate.loc[aggregate["algorithm"] == algorithm].sort_values(
            "scheduled_env_step"
        )
        steps = frame["scheduled_env_step"].to_numpy()
        means = frame["mean_eval_return"].to_numpy()
        spread = frame["std_across_seed_means"].to_numpy()
        axis.plot(steps, means, label=f"{algorithm} (n={int(frame['n_seeds'].iloc[0])})", color=colors[algorithm], linewidth=2)
        axis.fill_between(steps, means - spread, means + spread, color=colors[algorithm], alpha=0.18)

    axis.set_xlabel("Training environment steps")
    axis.set_ylabel("Deterministic evaluation return")
    axis.set_title("DQN Variant Comparison on the Same Evaluation Seed Set")
    axis.grid(True, linestyle="--", alpha=0.45)
    axis.legend(loc="best")
    _save_figure(figure, curve_path)

    final_path = save_dir / "final_evaluation_distribution.png"
    figure, axis = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(algorithms))
    for position, algorithm in enumerate(algorithms):
        values = auc_summary.loc[
            auc_summary["algorithm"] == algorithm, "final_eval_return"
        ].to_numpy()
        offsets = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else np.array([0.0])
        axis.scatter(
            np.full(len(values), position) + offsets,
            values,
            color=colors[algorithm],
            alpha=0.85,
            zorder=3,
        )
        axis.errorbar(
            position,
            values.mean(),
            yerr=values.std(ddof=1) if len(values) > 1 else 0.0,
            color="black",
            capsize=5,
            linewidth=1.5,
            zorder=4,
        )
    axis.set_xticks(positions, algorithms, rotation=25, ha="right")
    axis.set_ylabel("Final deterministic evaluation return")
    axis.set_title("Final Evaluation by Training Seed")
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    _save_figure(figure, final_path)

    auc_path_figure = save_dir / "learning_auc_summary.png"
    figure, axis = plt.subplots(figsize=(11, 6.5))
    auc_means = []
    auc_errors = []
    for algorithm in algorithms:
        values = auc_summary.loc[
            auc_summary["algorithm"] == algorithm, "normalized_eval_auc"
        ].dropna().to_numpy()
        auc_means.append(values.mean() if len(values) else np.nan)
        auc_errors.append(values.std(ddof=1) if len(values) > 1 else 0.0)
    axis.bar(positions, auc_means, yerr=auc_errors, color=[colors[name] for name in algorithms], capsize=5)
    axis.set_xticks(positions, algorithms, rotation=25, ha="right")
    axis.set_ylabel("Normalized evaluation AUC")
    axis.set_title("Learning Efficiency Summary")
    axis.grid(axis="y", linestyle="--", alpha=0.45)
    _save_figure(figure, auc_path_figure)

    training_curve_path = plot_training_convergence(
        experiment_dir, save_dir, algorithms, colors
    )
    training_overlay_path = plot_training_convergence_overlay(
        experiment_dir, save_dir, algorithms, colors
    )

    return {
        "aggregate_csv": aggregate_path,
        "auc_csv": auc_path,
        "evaluation_curve": curve_path,
        "final_distribution": final_path,
        "auc_figure": auc_path_figure,
        "training_convergence": training_curve_path,
        "training_convergence_overlay": training_overlay_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="繪製 DQN 的 deterministic evaluation 與訓練收斂診斷圖"
    )
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--save-dir", default=None)
    plot_mode = parser.add_mutually_exclusive_group()
    plot_mode.add_argument(
        "--training-only",
        action="store_true",
        help="只產生訓練收斂診斷圖，不覆寫既有 evaluation 圖表。",
    )
    plot_mode.add_argument(
        "--training-overlay-only",
        action="store_true",
        help="只產生將各模型收斂實線疊加的圖，不覆寫既有 evaluation 圖表。",
    )
    args = parser.parse_args()
    if args.training_only:
        path = plot_training_convergence_only(args.experiment_dir, args.save_dir)
        print(f"training_convergence: {path}")
    elif args.training_overlay_only:
        path = plot_training_convergence_overlay_only(args.experiment_dir, args.save_dir)
        print(f"training_convergence_overlay: {path}")
    else:
        paths = plot_experiment(args.experiment_dir, args.save_dir)
        for label, path in paths.items():
            print(f"{label}: {path}")
