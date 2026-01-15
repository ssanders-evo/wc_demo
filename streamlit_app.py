import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dataclasses import dataclass
from typing import Literal, Dict, Any
from scipy import stats


SelectionRule = Literal[
    "max_among_significant",
    "max_overall",
    "max_overall_then_check",
]

EffectScale = Literal[
    "risk_diff",
    "relative",
]


@dataclass(frozen=True)
class WinnerCurseConfig:
    n_variants: int = 10
    n_sims: int = 100_000
    alpha: float = 0.05
    two_sided: bool = True
    selection_rule: SelectionRule = "max_among_significant"

    n_control: int = 1000
    n_treat: int = 1000

    p_control: float = 0.10
    effect_scale: EffectScale = "risk_diff"
    true_effect: float = 0.005

    seed: int = 0
    continuity_correction: bool = False


def _validate_config(cfg: WinnerCurseConfig) -> None:
    if cfg.n_variants < 1:
        raise ValueError("n_variants must be >= 1")
    if cfg.n_sims < 1:
        raise ValueError("n_sims must be >= 1")
    if not (0 < cfg.alpha < 1):
        raise ValueError("alpha must be in (0, 1)")
    if cfg.n_control < 2 or cfg.n_treat < 2:
        raise ValueError("n_control and n_treat must be >= 2")
    if not (0 < cfg.p_control < 1):
        raise ValueError("p_control must be in (0, 1)")

    if cfg.effect_scale == "risk_diff":
        p_treat = cfg.p_control + cfg.true_effect
    else:
        p_treat = cfg.p_control * (1.0 + cfg.true_effect)

    if not (0 < p_treat < 1):
        raise ValueError(
            f"Implied p_treat={p_treat:.4f} is outside (0,1). "
            "Adjust p_control/true_effect/effect_scale."
        )


def _two_proportion_ztest_pvalue(
    x_t: np.ndarray, n_t: int,
    x_c: np.ndarray, n_c: int,
    two_sided: bool = True,
    continuity_correction: bool = True,
) -> np.ndarray:
    p_t = x_t / n_t
    p_c = x_c / n_c
    diff = p_t - p_c

    p_pool = (x_t + x_c) / (n_t + n_c)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
    se = np.where(se == 0, np.nan, se)

    if continuity_correction:
        cc = 0.5 * (1 / n_t + 1 / n_c)
        abs_diff = np.abs(diff)
        diff_adj_abs = np.maximum(0, abs_diff - cc)
        z = (diff_adj_abs * np.sign(diff)) / se
    else:
        z = diff / se

    if two_sided:
        pvals = 2 * stats.norm.sf(np.abs(z))
    else:
        pvals = stats.norm.sf(z)

    return np.where(np.isnan(pvals), 1.0, pvals)


def simulate_winners_curse_binary(cfg: WinnerCurseConfig) -> Dict[str, Any]:
    _validate_config(cfg)
    rng = np.random.default_rng(cfg.seed)

    K = cfg.n_variants
    S = cfg.n_sims
    n_c = cfg.n_control
    n_t = cfg.n_treat
    p_c = cfg.p_control

    if cfg.effect_scale == "risk_diff":
        p_t = p_c + cfg.true_effect
        true_rd = cfg.true_effect
        true_rel = (p_t / p_c - 1.0) if p_c > 0 else np.nan
    else:
        p_t = p_c * (1.0 + cfg.true_effect)
        true_rd = p_t - p_c
        true_rel = cfg.true_effect

    x_c = rng.binomial(n=n_c, p=p_c, size=(S, K))
    x_t = rng.binomial(n=n_t, p=p_t, size=(S, K))

    rd_hat = (x_t / n_t) - (x_c / n_c)

    p_c_hat = x_c / n_c
    p_t_hat = x_t / n_t
    rel_hat = np.where(p_c_hat == 0, np.nan, (p_t_hat / p_c_hat) - 1.0)

    pvals = _two_proportion_ztest_pvalue(
        x_t=x_t, n_t=n_t,
        x_c=x_c, n_c=n_c,
        two_sided=cfg.two_sided,
        continuity_correction=cfg.continuity_correction,
    )
    sig = pvals < cfg.alpha

    winner_idx = np.full(S, fill_value=-1, dtype=int)
    winner_rd = np.full(S, fill_value=np.nan, dtype=float)
    winner_rel = np.full(S, fill_value=np.nan, dtype=float)
    winner_p = np.full(S, fill_value=np.nan, dtype=float)

    if cfg.selection_rule == "max_among_significant":
        masked = np.where(sig, rd_hat, -np.inf)
        idx = np.argmax(masked, axis=1)
        best_val = masked[np.arange(S), idx]
        has_winner = np.isfinite(best_val)
        winner_idx[has_winner] = idx[has_winner]
        winner_rd[has_winner] = rd_hat[np.arange(S)[has_winner], idx[has_winner]]
        winner_rel[has_winner] = rel_hat[np.arange(S)[has_winner], idx[has_winner]]
        winner_p[has_winner] = pvals[np.arange(S)[has_winner], idx[has_winner]]

    elif cfg.selection_rule == "max_overall":
        idx = np.argmax(rd_hat, axis=1)
        winner_idx[:] = idx
        winner_rd[:] = rd_hat[np.arange(S), idx]
        winner_rel[:] = rel_hat[np.arange(S), idx]
        winner_p[:] = pvals[np.arange(S), idx]

    elif cfg.selection_rule == "max_overall_then_check":
        idx = np.argmax(rd_hat, axis=1)
        is_sig = sig[np.arange(S), idx]
        winner_idx[is_sig] = idx[is_sig]
        winner_rd[is_sig] = rd_hat[np.arange(S)[is_sig], idx[is_sig]]
        winner_rel[is_sig] = rel_hat[np.arange(S)[is_sig], idx[is_sig]]
        winner_p[is_sig] = pvals[np.arange(S)[is_sig], idx[is_sig]]

    else:
        raise ValueError(f"Unknown selection_rule: {cfg.selection_rule}")

    rd_sig_all = rd_hat[sig]
    rel_sig_all = rel_hat[sig]

    return {
        "cfg": cfg,
        "p_treat": float(p_t),
        "true_rd": float(true_rd),
        "true_rel": float(true_rel),
        "rd_hat": rd_hat,
        "rel_hat": rel_hat,
        "pvals": pvals,
        "sig": sig,
        "winner_idx": winner_idx,
        "winner_rd": winner_rd,
        "winner_rel": winner_rel,
        "winner_p": winner_p,
        "rd_sig_all": rd_sig_all,
        "rel_sig_all": rel_sig_all,
        "any_sig_rate": float(np.mean(np.any(sig, axis=1))),
        "winner_exists_rate": float(np.mean(np.isfinite(winner_rd))),
    }


def build_summary_table(res: Dict[str, Any]) -> pd.DataFrame:
    """
    Decision-oriented summary (no config block; inputs are visible in the sidebar).
    """
    true_rd = res["true_rd"]
    true_rel = res["true_rel"]

    sig_rd = res["rd_sig_all"][np.isfinite(res["rd_sig_all"])]
    win_rd = res["winner_rd"][np.isfinite(res["winner_rd"])]

    sig_rel = res["rel_sig_all"][np.isfinite(res["rel_sig_all"])]
    win_rel = res["winner_rel"][np.isfinite(res["winner_rel"])]

    def safe_mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else np.nan

    def safe_ratio(num: float, den: float) -> float:
        if np.isnan(num) or np.isnan(den) or den == 0:
            return np.nan
        return num / den

    assumed_rd_sig = safe_mean(sig_rd)
    assumed_rd_win = safe_mean(win_rd)

    assumed_rel_sig = safe_mean(sig_rel)
    assumed_rel_win = safe_mean(win_rel)

    rows = [
        ("Rates", "Any significant in portfolio", f"{res['any_sig_rate']:.3f}"),
        ("Rates", "Winner exists under rule", f"{res['winner_exists_rate']:.3f}"),

        ("Absolute (risk diff)", "True effect", f"{100*true_rd:.3f} pp"),
        ("Absolute (risk diff)", "Assumed effect (mean | significant)", f"{100*assumed_rd_sig:.3f} pp"),
        ("Absolute (risk diff)", "Inflation ratio (significant-only)", f"{safe_ratio(assumed_rd_sig, true_rd):.2f}x"),
        ("Absolute (risk diff)", "Assumed shipped effect (mean | winner)", f"{100*assumed_rd_win:.3f} pp"),
        ("Absolute (risk diff)", "Inflation ratio (winner)", f"{safe_ratio(assumed_rd_win, true_rd):.2f}x"),

        ("Relative lift", "True effect", f"{100*true_rel:.2f}%"),
        ("Relative lift", "Assumed effect (mean | significant)", f"{100*assumed_rel_sig:.2f}%"),
        ("Relative lift", "Inflation ratio (significant-only)", f"{safe_ratio(assumed_rel_sig, true_rel):.2f}x"),
        ("Relative lift", "Assumed shipped effect (mean | winner)", f"{100*assumed_rel_win:.2f}%"),
        ("Relative lift", "Inflation ratio (winner)", f"{safe_ratio(assumed_rel_win, true_rel):.2f}x"),

        ("Counts", "Significant variant results pooled", f"{sig_rd.size}"),
        ("Counts", "Winners (portfolios with a winner)", f"{win_rd.size}"),
    ]

    df = pd.DataFrame(rows, columns=["Section", "Metric", "Value"])
    return df


def make_figures(res: Dict[str, Any], bins: int = 120, fig_w: float = 6.0, fig_h: float = 3.4):
    """
    Smaller figures for a two-column Streamlit layout.
    True-effect line uses a non-blue color to avoid confusion with distributions.
    """
    true_rd = res["true_rd"]
    true_rel = res["true_rel"]

    all_rd = res["rd_hat"].ravel()
    all_rel = res["rel_hat"].ravel()
    all_rd = all_rd[np.isfinite(all_rd)]
    all_rel = all_rel[np.isfinite(all_rel)]

    sig_rd = res["rd_sig_all"][np.isfinite(res["rd_sig_all"])]
    win_rd = res["winner_rd"][np.isfinite(res["winner_rd"])]

    sig_rel = res["rel_sig_all"][np.isfinite(res["rel_sig_all"])]
    win_rel = res["winner_rel"][np.isfinite(res["winner_rel"])]

    # Risk difference figure
    fig1 = plt.figure(figsize=(fig_w, fig_h))
    if all_rd.size:
        plt.hist(all_rd, bins=bins, density=True, alpha=0.25, color="lightgray",
                 label="All estimates (unconditional)")
    if sig_rd.size:
        plt.hist(sig_rd, bins=bins, density=True, histtype="step", linewidth=2,
                 label="Significant-only")
    if win_rd.size:
        plt.hist(win_rd, bins=bins, density=True, histtype="step", linewidth=2,
                 label="Winner-selected")
    plt.axvline(true_rd, linewidth=2, color="black", linestyle="--", label="True effect (RD)")
    plt.title("Risk Difference")
    plt.xlabel("Risk difference (treat - control)")
    plt.ylabel("Density")
    plt.legend(fontsize=9)
    plt.tight_layout()

    # Relative lift figure
    fig2 = plt.figure(figsize=(fig_w, fig_h))
    if all_rel.size:
        plt.hist(all_rel, bins=bins, density=True, alpha=0.25, color="lightgray",
                 label="All estimates (unconditional)")
    if sig_rel.size:
        plt.hist(sig_rel, bins=bins, density=True, histtype="step", linewidth=2,
                 label="Significant-only")
    if win_rel.size:
        plt.hist(win_rel, bins=bins, density=True, histtype="step", linewidth=2,
                 label="Winner-selected")
    plt.axvline(true_rel, linewidth=2, color="black", linestyle="--", label="True effect (relative)")
    plt.title("Relative Lift")
    plt.xlabel("Relative lift ((p_t/p_c)-1)")
    plt.ylabel("Density")
    plt.legend(fontsize=9)
    plt.tight_layout()

    return fig1, fig2


@st.cache_data(show_spinner=False)
def run_simulation_cached(cfg: WinnerCurseConfig) -> Dict[str, Any]:
    return simulate_winners_curse_binary(cfg)


def main():
    st.set_page_config(page_title="Winner's Curse Simulator", layout="wide")
    st.title("Winner’s Curse in A/B/N Experimentation (Binary Outcome Simulator)")

    st.write(
        "Simulates an A/B/N test with K treatment variants vs a shared control, then shows how "
        "selection (significant-only and winner-picked) inflates observed lift."
    )

    with st.sidebar:
        st.header("Simulation Controls")

        n_variants = st.number_input("Number of treatments (K)", min_value=1, max_value=100, value=10, step=1)
        n_sims = st.number_input(
            "Number of portfolios (simulations)",
            min_value=1_000,
            max_value=500_000,
            value=100_000,
            step=1_000
        )
        alpha = st.slider("alpha", min_value=0.001, max_value=0.20, value=0.05, step=0.001)

        two_sided = st.checkbox("Two-sided test", value=True)

        selection_rule = st.selectbox(
            "Winner selection rule",
            options=["max_among_significant", "max_overall_then_check", "max_overall"],
            index=0
        )

        st.divider()
        st.subheader("Experiment Sizing")
        n_control = st.number_input("n_control per variant", min_value=10, max_value=1_000_000, value=1000, step=10)
        n_treat = st.number_input("n_treat per variant", min_value=10, max_value=1_000_000, value=1000, step=10)

        st.divider()
        st.subheader("Outcome Model")
        p_control = st.slider("Baseline conversion p_control", min_value=0.001, max_value=0.999, value=0.10, step=0.001)

        effect_scale = st.selectbox("Effect scale", options=["risk_diff", "relative"], index=0)

        if effect_scale == "risk_diff":
            true_effect = st.slider("True effect (absolute lift, in pp)", min_value=-0.20, max_value=0.20, value=0.005, step=0.001)
        else:
            true_effect = st.slider("True effect (relative lift)", min_value=-0.90, max_value=2.00, value=0.05, step=0.01)

        st.divider()
        st.subheader("Test Details")
        continuity_correction = st.checkbox("Continuity correction", value=False)
        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=0, step=1)

        st.divider()
        st.subheader("Display")
        bins = st.slider("Histogram bins", min_value=40, max_value=250, value=120, step=5)
        fig_w = st.slider("Figure width", min_value=4.0, max_value=10.0, value=6.0, step=0.5)
        fig_h = st.slider("Figure height", min_value=2.5, max_value=6.0, value=3.4, step=0.1)

    cfg = WinnerCurseConfig(
        n_variants=int(n_variants),
        n_sims=int(n_sims),
        alpha=float(alpha),
        two_sided=bool(two_sided),
        selection_rule=selection_rule,  # type: ignore
        n_control=int(n_control),
        n_treat=int(n_treat),
        p_control=float(p_control),
        effect_scale=effect_scale,      # type: ignore
        true_effect=float(true_effect),
        seed=int(seed),
        continuity_correction=bool(continuity_correction),
    )

    try:
        res = run_simulation_cached(cfg)
    except Exception as e:
        st.error(f"Simulation error: {e}")
        return

    table = build_summary_table(res)
    fig_rd, fig_rel = make_figures(res, bins=int(bins), fig_w=float(fig_w), fig_h=float(fig_h))

    # Two column layout:
    # - Left: summary table (and allow it to extend vertically)
    # - Right: plots
    c1, c2 = st.columns([1, 2], gap="large")

    with c1:
        st.subheader("Summary")
        # Make the table "taller" and fit the column width
        st.dataframe(
            table,
            use_container_width=True,
            hide_index=True,
            height=720
        )

    with c2:
        st.subheader("Distributions")
        st.pyplot(fig_rd, clear_figure=True)
        st.pyplot(fig_rel, clear_figure=True)

    with st.expander("How to interpret the key rows"):
        st.write(
            "• **Assumed effect (mean | significant)**: what you would believe if you only looked at statistically significant variants.\n\n"
            "• **Inflation ratio (significant-only)**: (mean | significant) ÷ true effect.\n\n"
            "• **Assumed shipped effect (mean | winner)**: what you would believe if you ship the selected winner.\n\n"
            "• **Inflation ratio (winner)**: (mean | winner) ÷ true effect.\n\n"
            "Gray is the unconditional sampling distribution; colored outlines are selected subsets."
        )


if __name__ == "__main__":
    main()

