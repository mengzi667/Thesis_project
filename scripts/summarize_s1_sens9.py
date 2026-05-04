from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    }
)


ROOT = Path(r"D:\TUD\Thesis_project")
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "s1_sens9_summary"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GroupCfg:
    gid: str
    w_l: float
    w_e: float
    beta_a: float
    beta_c: float
    beta_r: float


GROUPS = [
    GroupCfg("g0", 0.5, 0.5, 0.40, 0.12, 0.005),
    GroupCfg("g1", 0.5, 0.5, 0.30, 0.12, 0.005),
    GroupCfg("g2", 0.5, 0.5, 0.50, 0.12, 0.005),
    GroupCfg("g3", 0.5, 0.5, 0.40, 0.08, 0.005),
    GroupCfg("g4", 0.5, 0.5, 0.40, 0.16, 0.005),
    GroupCfg("g5", 0.7, 0.3, 0.40, 0.12, 0.005),
    GroupCfg("g6", 0.6, 0.4, 0.40, 0.12, 0.005),
    GroupCfg("g7", 0.4, 0.6, 0.40, 0.12, 0.005),
    GroupCfg("g8", 0.3, 0.7, 0.40, 0.12, 0.005),
]

PROFILES = ["main"]


def _save(fig: plt.Figure, name: str) -> None:
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def _load_eval(group_id: str, profile: str) -> pd.DataFrame:
    metrics_dir = RESULTS / f"s1_sens9_{group_id}_eval_{profile}" / "metrics"
    path_ckpt = metrics_dir / "eval_checkpoint_only.csv"
    path_three = metrics_dir / "eval_three_policies.csv"
    if path_ckpt.exists():
        return pd.read_csv(path_ckpt)
    if path_three.exists():
        return pd.read_csv(path_three)
    raise FileNotFoundError(f"Missing eval file under: {metrics_dir}")


def _build_long_df() -> pd.DataFrame:
    rows = []
    for g in GROUPS:
        for profile in PROFILES:
            df = _load_eval(g.gid, profile)
            ckpt_only_mode = "mean_reward" in df.columns

            if ckpt_only_mode:
                ck = {
                    "ckpt_mean_reward": float(df["mean_reward"].mean()),
                    "ckpt_std_reward": float(df["mean_reward"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_delta_edl": float(df["mean_delta_edl"].mean()),
                    "ckpt_std_delta_edl": float(df["mean_delta_edl"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_service_rate": float(df["service_rate"].mean()),
                    "ckpt_std_service_rate": float(df["service_rate"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_offers": float(df["relocation_offers"].mean()),
                    "ckpt_mean_realized_loss": float(df["mean_realized_loss"].mean()),
                    "ckpt_mean_reward_realized_term": float(df["mean_reward_realized_term"].mean()),
                    "ckpt_mean_reward_edl_term": float(df["mean_reward_edl_term"].mean()),
                    "ckpt_mean_reward_accept_term": float(df["mean_reward_accept_term"].mean()),
                }
                delta_reward_vs_no = float("nan")
                delta_reward_vs_ao = float("nan")
                delta_sr_vs_no = float("nan")
                delta_deltaedl_vs_ao = float("nan")
            else:
                ck = {
                    "ckpt_mean_reward": float(df["ckpt_mean_reward"].mean()),
                    "ckpt_std_reward": float(df["ckpt_mean_reward"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_delta_edl": float(df["ckpt_mean_delta_edl"].mean()),
                    "ckpt_std_delta_edl": float(df["ckpt_mean_delta_edl"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_service_rate": float(df["ckpt_service_rate"].mean()),
                    "ckpt_std_service_rate": float(df["ckpt_service_rate"].std(ddof=1)) if len(df) > 1 else 0.0,
                    "ckpt_mean_offers": float(df["ckpt_relocation_offers"].mean()),
                    "ckpt_mean_realized_loss": float(df["ckpt_mean_realized_loss"].mean()),
                    "ckpt_mean_reward_realized_term": float(df["ckpt_mean_reward_realized_term"].mean()),
                    "ckpt_mean_reward_edl_term": float(df["ckpt_mean_reward_edl_term"].mean()),
                    "ckpt_mean_reward_accept_term": float(df["ckpt_mean_reward_accept_term"].mean()),
                }
                no_reward = float(df["no_mean_reward"].mean())
                ao_reward = float(df["ao_mean_reward"].mean())
                no_sr = float(df["no_service_rate"].mean())
                ao_delta = float(df["ao_mean_delta_edl"].mean())
                delta_reward_vs_no = ck["ckpt_mean_reward"] - no_reward
                delta_reward_vs_ao = ck["ckpt_mean_reward"] - ao_reward
                delta_sr_vs_no = ck["ckpt_mean_service_rate"] - no_sr
                delta_deltaedl_vs_ao = ck["ckpt_mean_delta_edl"] - ao_delta

            rows.append(
                {
                    "group_id": g.gid,
                    "profile": profile,
                    "w_l": g.w_l,
                    "w_e": g.w_e,
                    "beta_a": g.beta_a,
                    "beta_c": g.beta_c,
                    "beta_r": g.beta_r,
                    **ck,
                    "delta_reward_vs_no": delta_reward_vs_no,
                    "delta_reward_vs_ao": delta_reward_vs_ao,
                    "delta_sr_vs_no": delta_sr_vs_no,
                    "delta_deltaedl_vs_ao": delta_deltaedl_vs_ao,
                }
            )
    return pd.DataFrame(rows)


def _plot_main_bars(main_df: pd.DataFrame) -> None:
    d = main_df.sort_values("group_id").copy()
    x = d["group_id"]

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.bar(
        x,
        d["ckpt_mean_reward"],
        color="#4C72B0",
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_title("Main Profile: Mean Reward by Group")
    ax.set_ylabel("Mean reward")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "main_ckpt_mean_reward")

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.bar(
        x,
        d["ckpt_mean_delta_edl"],
        color="#55A868",
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_title("Main Profile: Mean Delta EDL by Group")
    ax.set_ylabel("Mean delta EDL")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "main_ckpt_mean_delta_edl")

    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.bar(
        x,
        d["ckpt_mean_service_rate"],
        color="#7F7F7F",
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_title("Main Profile: Service Rate by Group")
    ax.set_ylabel("Service rate")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "main_ckpt_mean_service_rate")


def _plot_weight_scan(main_df: pd.DataFrame) -> None:
    w_df = main_df[main_df["group_id"].isin(["g5", "g6", "g7", "g8"])].copy()
    w_df["w_tag"] = w_df["w_l"].map(lambda x: f"{x:.1f}") + "/" + w_df["w_e"].map(lambda x: f"{x:.1f}")
    w_df = w_df.sort_values("w_l", ascending=False)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(
        w_df["w_tag"],
        w_df["ckpt_mean_reward"],
        color="#4C72B0",
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_title("Weight Scan (G5-G8): Mean Reward in Main Profile")
    ax.set_ylabel("Mean reward")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "weight_scan_main_reward_only")

    comp = w_df[
        [
            "w_tag",
            "ckpt_mean_reward_realized_term",
            "ckpt_mean_reward_edl_term",
            "ckpt_mean_reward_accept_term",
        ]
    ].copy()
    comp = comp.set_index("w_tag")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = range(len(comp.index))
    width = 0.25
    ax.bar(
        [i - width for i in x],
        comp["ckpt_mean_reward_realized_term"],
        width=width,
        color="#C44E52",
        label="Realized term",
    )
    ax.bar(
        x,
        comp["ckpt_mean_reward_edl_term"],
        width=width,
        color="#55A868",
        label="EDL term",
    )
    ax.bar(
        [i + width for i in x],
        comp["ckpt_mean_reward_accept_term"],
        width=width,
        color="#8172B3",
        label="Accept term",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), list(comp.index))
    ax.set_title("Weight Scan (G5-G8): Reward Component Means")
    ax.set_ylabel("Component mean")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _save(fig, "weight_scan_main_components")


def _plot_beta_scan(main_df: pd.DataFrame) -> None:
    b_df = main_df[main_df["group_id"].isin(["g0", "g1", "g2", "g3", "g4"])].copy()
    b_df["tag"] = b_df.apply(lambda r: f"a{r.beta_a:.2f}|c{r.beta_c:.2f}", axis=1)
    b_df = b_df.sort_values(["beta_a", "beta_c"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.bar(
        b_df["tag"],
        b_df["ckpt_mean_reward"],
        color="#4C72B0",
        linewidth=0.6,
        edgecolor="white",
    )
    ax.set_title("Beta Scan (G0-G4): Mean Reward in Main Profile")
    ax.set_ylabel("Mean reward")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    _save(fig, "beta_scan_main_reward_only")

    comp = b_df[
        [
            "tag",
            "ckpt_mean_reward_realized_term",
            "ckpt_mean_reward_edl_term",
            "ckpt_mean_reward_accept_term",
        ]
    ].copy()
    comp = comp.set_index("tag")
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    x = range(len(comp.index))
    width = 0.25
    ax.bar(
        [i - width for i in x],
        comp["ckpt_mean_reward_realized_term"],
        width=width,
        color="#C44E52",
        label="Realized term",
    )
    ax.bar(
        x,
        comp["ckpt_mean_reward_edl_term"],
        width=width,
        color="#55A868",
        label="EDL term",
    )
    ax.bar(
        [i + width for i in x],
        comp["ckpt_mean_reward_accept_term"],
        width=width,
        color="#8172B3",
        label="Accept term",
    )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(list(x), list(comp.index))
    ax.set_title("Beta Scan (G0-G4): Reward Component Means")
    ax.set_ylabel("Component mean")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _save(fig, "beta_scan_main_components")


def _plot_heatmap(main_df: pd.DataFrame) -> None:
    kpis = [
        "ckpt_mean_reward",
        "ckpt_mean_delta_edl",
        "ckpt_mean_service_rate",
        "ckpt_mean_offers",
        "ckpt_mean_realized_loss",
    ]
    hm = main_df.set_index("group_id")[kpis].copy()
    hm = (hm - hm.mean()) / (hm.std(ddof=0).replace(0.0, 1.0))

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    im = ax.imshow(hm.values, aspect="auto", cmap="RdYlBu_r")
    ax.set_yticks(range(len(hm.index)), hm.index)
    ax.set_xticks(range(len(hm.columns)), hm.columns, rotation=25, ha="right")
    ax.set_title("Sensitivity Heatmap (Main Profile, z-scored KPI)")
    plt.colorbar(im, ax=ax, shrink=0.85, label="z-score (group-standardized)")
    _save(fig, "kpi_heatmap_main")


def main() -> None:
    long_df = _build_long_df()
    long_df.to_csv(OUT_DIR / "all_groups_long.csv", index=False)

    main_df = long_df[long_df["profile"] == "main"].copy()
    main_rank = main_df.sort_values("ckpt_mean_reward", ascending=False)
    main_rank.to_csv(OUT_DIR / "main_profile_rank.csv", index=False)

    _plot_main_bars(main_df.sort_values("group_id"))
    _plot_weight_scan(main_df)
    _plot_beta_scan(main_df)
    _plot_heatmap(main_df.sort_values("group_id"))
    # main-only sensitivity run: no cross-profile robustness plot

    print(f"Wrote: {OUT_DIR / 'all_groups_long.csv'}")
    print(f"Wrote: {OUT_DIR / 'main_profile_rank.csv'}")
    print("Wrote main-only sensitivity figure set in results/s1_sens9_summary")


if __name__ == "__main__":
    main()
