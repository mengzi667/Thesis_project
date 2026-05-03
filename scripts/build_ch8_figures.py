from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(r"D:\TUD\Thesis_project")
FIG_DIR = ROOT / r"docs\_tmp_thesis_scan\thesis_template\figures"


def _load_eval_summary(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    return df.iloc[0]


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def fig_6_4_reward_curve(train_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    x = train_metrics["episode"]
    y = train_metrics["mean_reward"]
    y_ma = y.rolling(20, min_periods=1).mean()
    ax.plot(x, y, color="#7aa6c2", alpha=0.35, linewidth=1.0, label="Episode reward")
    ax.plot(x, y_ma, color="#1f4e79", linewidth=2.0, label="Moving average (20)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean reward")
    ax.set_title("Scenario 1 Training Reward Curve")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "ch8_fig_6_4_s1_train_reward_curve")


def fig_6_5_loss_epsilon(train_metrics: pd.DataFrame, loss_curve: pd.DataFrame) -> None:
    loss_ep = loss_curve.groupby("episode", as_index=False)["loss"].mean()

    fig, ax1 = plt.subplots(figsize=(6.6, 3.6))
    ax2 = ax1.twinx()

    ax1.plot(loss_ep["episode"], loss_ep["loss"], color="#8c4f91", linewidth=1.6, label="Loss (episode mean)")
    ax2.plot(train_metrics["episode"], train_metrics["epsilon"], color="#1b9e77", linewidth=1.6, label="Epsilon")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Epsilon")
    ax1.set_title("Training Loss and Epsilon Decay")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=8, loc="upper right")
    _save(fig, "ch8_fig_6_5_s1_train_loss_epsilon")


def fig_6_6_offers_accept(train_metrics: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(6.8, 3.9))
    ax2 = ax1.twinx()

    offers_ma = train_metrics["relocation_offers"].rolling(20, min_periods=1).mean()
    acc_ma = train_metrics["accept_rate"].rolling(20, min_periods=1).mean()

    ax1.plot(train_metrics["episode"], offers_ma, color="#d95f02", linewidth=2.0, label="Offers (MA20)")
    ax2.plot(train_metrics["episode"], acc_ma, color="#1b7837", linewidth=2.0, label="Acceptance rate (MA20)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Offers")
    ax2.set_ylabel("Acceptance rate")
    ax1.set_title("Offer Activation and Acceptance During Training")
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        frameon=False,
        fontsize=8,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.17),
    )
    fig.subplots_adjust(top=0.82)
    _save(fig, "ch8_fig_6_6_s1_train_offers_acceptance")


def fig_6_7_main_kpi(main: pd.Series) -> None:
    policies = ["No-offer", "Always-offer", "Checkpoint"]
    service = [main["no_mean_service_rate"], main["ao_mean_service_rate"], main["ckpt_mean_service_rate"]]
    offers = [main["no_mean_offers"], main["ao_mean_offers"], main["ckpt_mean_offers"]]
    accepted = [0.0, main["ao_mean_offers"], main["ckpt_mean_offers"] if main["ckpt_mean_offers"] == 0 else main["ckpt_mean_offers"] * (main["ckpt_sum_reward_accept_term"] / max(1e-12, 0.4 * 400.0))]
    # Acceptance rate derived from eval summary means:
    # no-offer: 0, always-offer: accepted/offers, checkpoint: accepted/offers.
    ar_no = 0.0
    ar_ao = (main["ao_mean_offers"] / main["ao_mean_offers"]) if main["ao_mean_offers"] > 0 else 0.0
    # ckpt accepted mean can be recovered approximately from accept term mean / beta_a (beta_a=0.4 in current runs)
    ckpt_acc_mean = main["ckpt_mean_reward_accept_term"] / 0.4 if 0.4 > 0 else 0.0
    ar_ckpt = (ckpt_acc_mean / main["ckpt_mean_offers"]) if main["ckpt_mean_offers"] > 0 else 0.0
    accept_rate = [ar_no, ar_ao, ar_ckpt]
    reward = [main["no_mean_reward"], main["ao_mean_reward"], main["ckpt_mean_reward"]]
    delta_edl = [main["no_mean_delta_edl"], main["ao_mean_delta_edl"], main["ckpt_mean_delta_edl"]]
    realized_loss = [main["no_mean_realized_loss"], main["ao_mean_realized_loss"], main["ckpt_mean_realized_loss"]]
    transitions = [main["no_mean_transitions"], main["ao_mean_transitions"], main["ckpt_mean_transitions"]]

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    colors = ["#7f7f7f", "#d95f02", "#1f78b4"]

    x = range(len(policies))
    # Panel A: service rate
    ax = axes[0, 0]
    ax.bar(x, service, color=colors, alpha=0.9)
    ax.set_xticks(list(x), policies, rotation=10)
    ax.set_ylim(0.9, 1.01)
    ax.set_ylabel("Service rate")
    ax.set_title("A. Service Rate")

    # Panel B: offer behavior
    ax = axes[0, 1]
    width = 0.26
    ax.bar([i - width for i in x], offers, width=width, color="#f16913", label="Offers")
    ax.bar(x, [0.0, main["ao_mean_offers"], ckpt_acc_mean], width=width, color="#31a354", label="Accepted")
    ax.bar([i + width for i in x], accept_rate, width=width, color="#756bb1", label="Acceptance rate")
    ax.set_xticks(list(x), policies, rotation=10)
    ax.set_title("B. Offer and Acceptance")
    ax.legend(frameon=False, fontsize=7)

    # Panel C: reward signals
    ax = axes[1, 0]
    width = 0.35
    ax.bar([i - width / 2 for i in x], reward, width=width, color="#4C78A8", label="Mean reward")
    ax.bar([i + width / 2 for i in x], delta_edl, width=width, color="#59A14F", label="Mean ΔEDL")
    ax.set_xticks(list(x), policies, rotation=10)
    ax.set_title("C. Reward and ΔEDL")
    ax.legend(frameon=False, fontsize=7)

    # Panel D: realized loss and transitions
    ax = axes[1, 1]
    width = 0.35
    ax.bar([i - width / 2 for i in x], realized_loss, width=width, color="#cb181d", label="Mean realized loss")
    ax.bar([i + width / 2 for i in x], transitions, width=width, color="#636363", label="Mean transitions")
    ax.set_xticks(list(x), policies, rotation=10)
    ax.set_title("D. Loss and Transition Density")
    ax.legend(frameon=False, fontsize=7)

    for ax in axes.ravel():
        ax.grid(axis="y", alpha=0.2, linestyle="--", linewidth=0.6)

    fig.suptitle("Main Profile KPI Comparison by Policy", y=0.99)
    fig.tight_layout()
    _save(fig, "ch8_fig_6_7_main_profile_kpi_comparison")


def fig_6_8_reward_edl_profiles(sparse: pd.Series, main: pd.Series, dense: pd.Series) -> None:
    profiles = ["Sparse", "Main", "Dense"]
    pols = [
        ("No-offer", "no", "#7f7f7f"),
        ("Always-offer", "ao", "#d95f02"),
        ("Checkpoint", "ckpt", "#1f78b4"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    data = [sparse, main, dense]

    for label, prefix, color in pols:
        axes[0].plot(profiles, [d[f"{prefix}_mean_reward"] for d in data], marker="o", linewidth=1.8, color=color, label=label)
        axes[1].plot(profiles, [d[f"{prefix}_mean_delta_edl"] for d in data], marker="o", linewidth=1.8, color=color, label=label)

    axes[0].set_title("Mean Reward")
    axes[1].set_title("Mean ΔEDL")
    axes[0].set_ylabel("Value")
    axes[0].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[1].grid(alpha=0.25, linestyle="--", linewidth=0.6)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Reward and EDL Robustness Across Profiles", y=1.02)
    fig.tight_layout()
    _save(fig, "ch8_fig_6_8_reward_edl_robustness")


def fig_6_9_offers_profiles(sparse: pd.Series, main: pd.Series, dense: pd.Series) -> None:
    profiles = ["Sparse", "Main", "Dense"]
    pols = [
        ("No-offer", "no", "#7f7f7f"),
        ("Always-offer", "ao", "#d95f02"),
        ("Checkpoint", "ckpt", "#1f78b4"),
    ]
    data = [sparse, main, dense]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for label, prefix, color in pols:
        ax.plot(profiles, [d[f"{prefix}_mean_offers"] for d in data], marker="o", linewidth=2.0, color=color, label=label)
    ax.set_title("Offer Activation Robustness Across Profiles")
    ax.set_ylabel("Mean offers")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "ch8_fig_6_9_offers_robustness")


def fig_6_10_reward_breakdown(main: pd.Series) -> None:
    policies = ["No-offer", "Always-offer", "Checkpoint"]
    realized = [main["no_mean_reward_realized_term"], main["ao_mean_reward_realized_term"], main["ckpt_mean_reward_realized_term"]]
    edl = [main["no_mean_reward_edl_term"], main["ao_mean_reward_edl_term"], main["ckpt_mean_reward_edl_term"]]
    acc = [main["no_mean_reward_accept_term"], main["ao_mean_reward_accept_term"], main["ckpt_mean_reward_accept_term"]]

    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    x = list(range(len(policies)))
    width = 0.24
    ax.bar([i - width for i in x], realized, width=width, color="#b2182b", alpha=0.9, label="Realized term")
    ax.bar(x, edl, width=width, color="#2166ac", alpha=0.9, label="EDL term")
    ax.bar([i + width for i in x], acc, width=width, color="#1a9850", alpha=0.9, label="Accept term")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, policies, rotation=10)
    ax.set_ylabel("Mean contribution")
    ax.set_title("Reward Component Breakdown (Main Profile)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    _save(fig, "ch8_fig_6_10_reward_component_breakdown")


def fig_6_2_trip_intensity(sparse: pd.Series, main: pd.Series, dense: pd.Series) -> None:
    profiles = ["Sparse", "Main", "Dense"]
    means = [sparse["ckpt_trip_count_mean"], main["ckpt_trip_count_mean"], dense["ckpt_trip_count_mean"]]
    variances = [sparse["ckpt_trip_count_var"], main["ckpt_trip_count_var"], dense["ckpt_trip_count_var"]]
    stds = [v ** 0.5 for v in variances]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    colors = ["#9ecae1", "#6baed6", "#3182bd"]
    ax.bar(profiles, means, color=colors, yerr=stds, capsize=4, alpha=0.9)
    ax.set_ylabel("Trip count mean")
    ax.set_title("Generated Trip Intensity Across Profiles")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    _save(fig, "ch8_fig_6_2_generated_trip_intensity")


def fig_6_3_opportunity_density(sparse: pd.Series, main: pd.Series, dense: pd.Series) -> None:
    profiles = ["Sparse", "Main", "Dense"]
    transitions = [sparse["ckpt_mean_transitions"], main["ckpt_mean_transitions"], dense["ckpt_mean_transitions"]]
    offers = [sparse["ckpt_mean_offers"], main["ckpt_mean_offers"], dense["ckpt_mean_offers"]]

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    x = list(range(len(profiles)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], transitions, width=width, color="#807dba", label="Decision transitions")
    ax.bar([i + width / 2 for i in x], offers, width=width, color="#f16913", label="Activated offers")
    ax.set_xticks(x, profiles)
    ax.set_ylabel("Count per episode (mean)")
    ax.set_title("Decision Transitions vs Activated Offers (Checkpoint)")
    ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "ch8_fig_6_3_offer_transition_density")


def main() -> None:
    train_metrics = pd.read_csv(ROOT / r"results\rl_s1_newmain_train\metrics\train_episode_metrics.csv")
    loss_curve = pd.read_csv(ROOT / r"results\rl_s1_newmain_train\metrics\train_loss_curve.csv")
    sparse = _load_eval_summary(ROOT / r"results\rl_s1_newmain_eval_sparse\metrics\eval_summary.csv")
    main_prof = _load_eval_summary(ROOT / r"results\rl_s1_newmain_eval_main\metrics\eval_summary.csv")
    dense = _load_eval_summary(ROOT / r"results\rl_s1_newmain_eval_dense\metrics\eval_summary.csv")

    fig_6_4_reward_curve(train_metrics)
    fig_6_5_loss_epsilon(train_metrics, loss_curve)
    fig_6_6_offers_accept(train_metrics)
    fig_6_2_trip_intensity(sparse, main_prof, dense)
    fig_6_3_opportunity_density(sparse, main_prof, dense)
    fig_6_7_main_kpi(main_prof)
    fig_6_8_reward_edl_profiles(sparse, main_prof, dense)
    fig_6_9_offers_profiles(sparse, main_prof, dense)
    fig_6_10_reward_breakdown(main_prof)
    print("Chapter 8 figures generated.")


if __name__ == "__main__":
    main()
