from matplotlib import pyplot as plt
import wandb
import numpy as np
import pandas as pd
import matplotlib.ticker as mticker

plt.rcParams.update({
    # send all text (including math) through LaTeX
    "text.usetex": True,
    "pgf.texsystem": "pdflatex",            # ensure it uses your TeX Live pdflatex
    # select the serif family, pointing to CM Roman
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    # embed actual Type-1 outlines (not bitmaps)
    "pdf.fonttype": 42,
    "ps.fonttype" : 42,
    # include any math packages you need
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}"
})

TIMESTEP_TO_PLOT = 900  # If a run contains fewer than this many logged steps, the last available step is used.
NO_DATA_FOR_LOSS_DIFFERENCE = 2048
WANDB_PROJECT = "balintszarvas-university-of-amsterdam"
WANDB_NAME = "clifford-equivariant-cnns"

# Fixed colors per model to ensure consistency across panels
MODEL_COLORS = {
    "Cond. CSCNN (Ours)": "royalblue",
    "CSCNN": "orangered",
    "Clifford ResNet": "mediumseagreen",
    "Steerable ResNet": "dodgerblue",
    "ResNet": "purple",
    "FNO": "gray",
    "G-FNO": "darkorange",
}

#Results for 2d Navier-Stokes experiments
samples = [64, 512, 2048, 4096]
resnet = dict(zip(samples, [0.004574, 0.003292, 0.002802, 0.002786]), Model='ResNet')
sresnet = dict(zip(samples, [0.008408, 0.002302, 0.001347, 0.001248]), Model='Steerable ResNet')
cresnet = dict(zip(samples, [0.004205, 0.002013, 0.001183, 0.001111]), Model='Clifford ResNet')
#gcresnet = dict(zip(samples, [0.002616, 0.001211, 0.0009139, 0.0008803]), Model='Clifford-Steerable ResNet')
fno = dict(zip(samples, [0.0317, 0.0116, 0.0087, 0.0082]), Model='FNO')
gfno = dict(zip(samples, [0.0187, 0.006775,  0.005411, 0.004903]), Model='G-FNO')

#wandb ids
ns_conditioned_64 = "cxntlqck"
ns_conditioned_128 = "6qclhjhg"
ns_conditioned_512 = "rkjgh1dx"
ns_conditioned_2048 = "w7zj3cu1"
ns_conditioned_4096 = "cejp5yta"


ns_default_64 = "hgxbov56"
ns_default_128 = "h436qtrf"
ns_default_512 = "64p70ykb"
ns_default_2048 = "rdxsh2mw"
ns_default_4096 = "kh3kxrbo"

result_df = pd.DataFrame([fno, gfno, cresnet, resnet, sresnet])


#Results for 3d Maxwell experiments
samples_mw3 = [64, 256, 512]

# Your models with their corresponding results (MW³)
cresnet_mw3 = dict(zip(samples_mw3, [0.003834, 0.002567, 0.001703]), Model='Clifford ResNet')
resnet_mw3 = dict(zip(samples_mw3, [0.003342, 0.001898, 0.001673]), Model='ResNet')
fno_mw3 = dict(zip(samples_mw3, [0.01907, 0.01664, 0.01415]), Model='FNO')

# Optional: if you have final values for these, declare them similarly
default_cscnn_mw3     = dict(zip(samples_mw3, [0.0007497, 0.00070165, 0.00068173]), Model='CSCNN')
conditioned_cscnn_mw3 = dict(zip(samples_mw3, [0.00060693, 0.00054376, 0.000451114]), Model='Cond. CSCNN (Ours)')

_mw3_series = [fno_mw3, cresnet_mw3, resnet_mw3]
try:
    _mw3_series.append(default_cscnn_mw3)  # noqa: F821
except NameError:
    pass
try:
    _mw3_series.append(conditioned_cscnn_mw3)  # noqa: F821
except NameError:
    pass

result_mw3_df = pd.DataFrame(_mw3_series)

#results for 2d relativistic Maxwell experiments
samples = [512, 1024, 2048]
resnet = dict(zip(samples, [0.80812, 0.7661,0.66529]), Model='ResNet')
gcresnet = dict(zip(samples, [0.38887, 0.3682,0.34944]), Model='CSCNN')
#condnet = dict(zip(samples, [0.20844, 0.20927]), Model='Conditioned Clifford-Steerable ResNet (Ours)')
#placeholder for the relativistic Maxwell experiment
condnet = dict(zip(samples, [0.3273, 0.29339, 0.2519]), Model='Cond. CSCNN (Ours)')



ns_conditioned_ids = [ns_conditioned_64, ns_conditioned_128, ns_conditioned_512,
           ns_conditioned_2048, ns_conditioned_4096]
ns_default_ids = [ns_default_64, ns_default_128, ns_default_512, ns_default_2048, ns_default_4096]
try:
    mw3_conditioned_ids = [mw3_conditioned_64, mw3_conditioned_256, mw3_conditioned_496]
except NameError:
    mw3_conditioned_ids = []
try:
    mw3_default_ids = [mw3_default_64, mw3_default_256, mw3_default_496]
except NameError:
    mw3_default_ids = []



api = wandb.Api()

def get_numsamples(id):
    run = api.run(f"{WANDB_PROJECT}/{WANDB_NAME}/{id}")
    numsamples = run.config["num_data"]
    return numsamples

def _get_metric_at_timestep(run, metric_key: str):
    """Return the metric value at the requested timestep.

    1. Try to locate the row whose logged **_step** equals `TIMESTEP_TO_PLOT`.
    2. If that exact step hasn't been logged, fall back to the final (last)
       value that *has* been logged.  This guarantees we compare the same
       timestep when it exists, while remaining robust to shorter runs.
    """

    df = run.history(keys=[metric_key, "_step"], pandas=True)

    if df.empty or metric_key not in df:
        return np.nan

    # Prefer the exact timestep if present ----------------------------------
    step_match = df.loc[df["_step"] == TIMESTEP_TO_PLOT]
    if not step_match.empty:
        return step_match[metric_key].dropna().iloc[-1]

    # Otherwise fall back to the last valid entry ---------------------------
    valid_series = df[metric_key].dropna()
    return valid_series.iloc[-1] if not valid_series.empty else np.nan


def load_runs(experiment_type: str):
    """Load MSE values and num_data for the requested experiment.

    Returns two dictionaries: (conditioned_runs, default_runs) where the keys
    are run-ids and the values are dicts with keys `total_mse`, `vector_mse`,
    `scalar_mse`, each mapped to [mse_value, num_samples].
    """

    if experiment_type == "ns":
        conditioned_ids = ns_conditioned_ids
        default_ids = ns_default_ids
    elif experiment_type == "mw3":
        conditioned_ids = mw3_conditioned_ids
        default_ids = mw3_default_ids
    else:
        raise ValueError("experiment_type must be either 'ns' or 'mw3'")

    conditioned_runs = {}
    default_runs = {}

    # Load conditioned runs
    for run_id in conditioned_ids:
        run = api.run(f"{WANDB_PROJECT}/{WANDB_NAME}/{run_id}")
        numsamples = get_numsamples(run_id)

        conditioned_runs[run_id] = {
            "total_mse":  [_get_metric_at_timestep(run, "test.loss_total"),  numsamples],
            "vector_mse": [_get_metric_at_timestep(run, "test.loss_vector"), numsamples],
            "scalar_mse": [_get_metric_at_timestep(run, "test.loss_scalar"), numsamples],
        }

    # Load default runs
    for run_id in default_ids:
        run = api.run(f"{WANDB_PROJECT}/{WANDB_NAME}/{run_id}")
        numsamples = get_numsamples(run_id)

        default_runs[run_id] = {
            "total_mse":  [_get_metric_at_timestep(run, "test.loss_total"),  numsamples],
            "vector_mse": [_get_metric_at_timestep(run, "test.loss_vector"), numsamples],
            "scalar_mse": [_get_metric_at_timestep(run, "test.loss_scalar"), numsamples],
        }

    return conditioned_runs, default_runs

def _extract_xy(runs_dict, ordered_ids, loss_key):
    """Return x (numsamples) and y (mse) arrays sorted by numsamples."""
    pts = [(runs_dict[rid][loss_key][1], runs_dict[rid][loss_key][0]) for rid in ordered_ids]
    pts.sort(key=lambda t: t[0])  # sort by num_data
    x = [p[0] for p in pts]
    y = [p[1] for p in pts]
    return x, y


def _plot_metric(experiment_type: str, loss_key: str, ylabel: str, title: str, ylim: tuple):
    conditioned_runs, default_runs = load_runs(experiment_type)

    if experiment_type == "ns":
        conditioned_ids = ns_conditioned_ids
        default_ids = ns_default_ids
    else:
        conditioned_ids = mw3_conditioned_ids
        default_ids = mw3_default_ids

    x_cond, y_cond = _extract_xy(conditioned_runs, conditioned_ids, loss_key)
    x_def,  y_def  = _extract_xy(default_runs,     default_ids,     loss_key)

    plt.figure(figsize=(8, 4))
    plt.plot(x_cond, y_cond, "o--", label="Cond. CSCNN (Ours)", color="royalblue")
    plt.plot(x_def,  y_def,  "o--", label="CSCNN", color="orangered")
    plt.yscale("log")
    plt.xlabel(r"No. of Simulations in Training Set")
    plt.ylabel(ylabel)
    #plt.title(title)
    plt.ylim(ylim)
    plt.legend()
    #plt.legend(
    #loc="upper center",          # which corner of the legend box is anchored
    #bbox_to_anchor=(1.02, 1),  # (x, y) in axes-fraction units (>1 puts it outside)
    #borderaxespad=0.0,
    #ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"figures/{experiment_type}_{loss_key}.pdf")


# ===== Navier–Stokes (NS) plots =====
def plot_ns_total_mse_vs_numsamples():
    _plot_metric("ns", "total_mse", r"MSE", r"Navier–Stokes R² – Total MSE", (1e-4, 2e-2))
    

def plot_ns_vector_mse_vs_numsamples():
    _plot_metric("ns", "vector_mse", r"MSE", r"Navier–Stokes R² – Vector MSE", (1e-4, 1e-2))
    

def plot_ns_scalar_mse_vs_numsamples():
    _plot_metric("ns", "scalar_mse", r"MSE", r"Navier–Stokes R² – Scalar MSE", (5e-4, 1e-2))


# ===== MW³ plots =====
def plot_mw3_total_mse_vs_numsamples():
    _plot_metric("mw3", "total_mse", r"MSE", r"MW³ – Total MSE", (1e-4, 1e-2))


def plot_mw3_vector_mse_vs_numsamples():
    _plot_metric("mw3", "vector_mse", r"MSE", r"MW³ – Vector MSE", (5e-4, 1e-2))


def plot_mw3_scalar_mse_vs_numsamples():
    _plot_metric("mw3", "scalar_mse", r"MSE", r"MW³ – Scalar MSE", (5e-4, 1e-2))


# ---------------------------------------------------------------------------
# BAR-PLOT OF LOSS DIFFERENCES (Default − Conditioned) AT A SPECIFIC DATA SIZE
# ---------------------------------------------------------------------------


def _find_run_id_by_samples(id_list, runs_dict, target_samples):
    """Return the first run-id whose stored num_samples equals target_samples."""
    for rid in id_list:
        if runs_dict[rid]["total_mse"][1] == target_samples:
            return rid
    raise ValueError(f"No run in list has num_data == {target_samples}")


def bar_plot():
    """Create a bar plot of (Default – Conditioned) MSE at `NO_DATA_FOR_LOSS_DIFFERENCE`.

    Uses the Navier–Stokes experiment by default because that dataset contains
    a run with the requested number of simulations (4096).
    """

    # Load NS runs -----------------------------------------------------------
    cond_runs, def_runs = load_runs("ns")

    # Identify the matching runs (same num_data) ----------------------------
    cond_id = _find_run_id_by_samples(ns_conditioned_ids, cond_runs, NO_DATA_FOR_LOSS_DIFFERENCE)
    def_id  = _find_run_id_by_samples(ns_default_ids,     def_runs, NO_DATA_FOR_LOSS_DIFFERENCE)

    # Compute differences ----------------------------------------------------
    total_diff  = def_runs[def_id]["total_mse"][0]  - cond_runs[cond_id]["total_mse"][0]
    vector_diff = def_runs[def_id]["vector_mse"][0] - cond_runs[cond_id]["vector_mse"][0]
    scalar_diff = def_runs[def_id]["scalar_mse"][0] - cond_runs[cond_id]["scalar_mse"][0]

    categories = ["Scalar", "Vector", "Total"]
    diffs      = [scalar_diff, vector_diff, total_diff]

    plt.figure(figsize=(4, 4))
    bars = plt.bar(categories, diffs, color=["royalblue", "orangered", "mediumseagreen"], alpha=0.7)
    plt.ylabel("Default – Conditioned MSE (↓)")
    #plt.title(f"MSE improvement at num_data = {NO_DATA_FOR_LOSS_DIFFERENCE}")

    # Annotate bars with values ---------------------------------------------
    for bar, val in zip(bars, diffs):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f"{val:.2e}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# BAR-PLOT OF PERCENTAGE IMPROVEMENT
# ---------------------------------------------------------------------------


def bar_plot_percent():
    """Bar chart of percentage MSE improvement (Default → Conditioned).

    Percentage is computed as `(default - conditioned) / default * 100` for the
    run whose `num_data` equals `NO_DATA_FOR_LOSS_DIFFERENCE`.
    """
    plt.rcParams.update({
        "font.size": 18,             # base font size
        "axes.labelsize": 18,        # x-/y-label size
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,

        })

    cond_runs, def_runs = load_runs("ns")

    cond_id = _find_run_id_by_samples(ns_conditioned_ids, cond_runs, NO_DATA_FOR_LOSS_DIFFERENCE)
    def_id  = _find_run_id_by_samples(ns_default_ids,     def_runs, NO_DATA_FOR_LOSS_DIFFERENCE)

    total_pct  = (def_runs[def_id]["total_mse"][0]  - cond_runs[cond_id]["total_mse"][0])  / def_runs[def_id]["total_mse"][0]  * 100
    vector_pct = (def_runs[def_id]["vector_mse"][0] - cond_runs[cond_id]["vector_mse"][0]) / def_runs[def_id]["vector_mse"][0] * 100
    scalar_pct = (def_runs[def_id]["scalar_mse"][0] - cond_runs[cond_id]["scalar_mse"][0]) / def_runs[def_id]["scalar_mse"][0] * 100

    categories = ["Scalar", "Vector", "Total"]
    pct_vals   = [scalar_pct, vector_pct, total_pct]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(categories, pct_vals,
                  color=["royalblue", "orangered", "mediumseagreen"],
                  alpha=0.7)

    # 1) add 15 % margin above the tallest bar ------------------------------
    ymax = max(pct_vals) * 1.15
    ax.set_ylim(0, ymax)

    # 2) annotate each bar with a slight offset -----------------------------
    for bar, val in zip(bars, pct_vals):
        ax.annotate(f"{val:.1f}%",
                    xy=(bar.get_x() + bar.get_width()/2, val),
                    xytext=(0, 3),                  # 3 points upward
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=18)

    ax.set_ylabel(r"MSE reduction (\%)")
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig("figures/percent_improvement.pdf")
    plt.show()


# ---------------------------------------------------------------------------
# SIDE-BY-SIDE TOTAL-MSE CURVES (NS vs MW³ vs Relativistic Maxwell)
# ---------------------------------------------------------------------------


def plot_total_mse_side_by_side(
    ns_ylim=(1.5e-4, 3e-2),
    mw_ylim=(2.5e-4, 3e-2),
    ns_yticks=None,
    mw_yticks=None,
    rel_ylim=(2e-1, 1.5),
    rel_yticks=None,
):
    """Plot NS, MW³ and relativistic Maxwell (R$^{1,2}$) total-MSE side by side.

    • common legend (top)
    • shared y-label
    • individual x-labels / ticks per subplot
    • overlays WANDB series and static declared baselines
    """

    plt.rcParams.update({
    "font.size": 25,             # base font size
    "axes.labelsize": 25,        # x-/y-label size
    "legend.fontsize": 25,
    "xtick.labelsize": 25,
    "ytick.labelsize": 25,
    })

    # independent y-axes so each subplot can have its own limits/ticks
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    # --- helper: draw static (declared-in-script) baselines ---------------
    def _draw_static_from_df(ax, df, linestyle='-', marker='s', linewidth=2, style_map=None):
        if df is None or len(df) == 0:
            return
        cols = [c for c in df.columns if c != "Model"]

        def _as_int(v):
            try:
                return int(v)
            except Exception:
                return v

        cols_sorted = sorted(cols, key=_as_int)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])

        for idx, (_, row) in enumerate(df.iterrows()):
            x_vals = []
            y_vals = []
            for c in cols_sorted:
                val = row.get(c)
                if pd.notna(val):
                    x_vals.append(_as_int(c))
                    y_vals.append(float(val))
            if not x_vals:
                continue
            label = row.get("Model", f"series_{idx}")
            color = MODEL_COLORS.get(label, (color_cycle[idx % len(color_cycle)] if color_cycle else None))
            # Optional style overrides per label (e.g., force scatter)
            style = style_map.get(label, {}) if style_map else {}
            if style.get("scatter"):
                ax.scatter(x_vals, y_vals, marker=style.get("marker", marker), label=label, color=color)
            else:
                ls = style.get("linestyle", linestyle)
                mk = style.get("marker", marker)
                lw = style.get("linewidth", linewidth)
                ax.plot(x_vals, y_vals, mk+ls, label=label, color=color, lw=lw)

    # --- helper: draw one panel (WANDB + static overlays) -----------------
    def _draw(ax, exp, ylim, yticks, static_df=None, style_map=None):
        cond_runs, def_runs = load_runs(exp)
        if exp == "ns":
            cond_ids = ns_conditioned_ids
            def_ids = ns_default_ids
        else:
            cond_ids = mw3_conditioned_ids
            def_ids = mw3_default_ids

        x_cond, y_cond = _extract_xy(cond_runs, cond_ids, "total_mse")
        x_def,  y_def  = _extract_xy(def_runs,  def_ids,  "total_mse")

        if len(x_cond) > 0:
            ax.plot(
                x_cond, y_cond, "s--",
                label="Cond. CSCNN (Ours)",
                color=MODEL_COLORS.get("Cond. CSCNN (Ours)", "royalblue"), lw=2,
            )
        if len(x_def) > 0:
            ax.plot(
                x_def, y_def, "s-",
                label="CSCNN",
                color=MODEL_COLORS.get("CSCNN", "orangered"), lw=2,
            )

        _draw_static_from_df(ax, static_df, style_map=style_map)

        ax.set_yscale("log")
        ax.set_ylim(ylim)
        ax.set_xlabel(r"No. of Training Simulations")
        ax.grid(True, which="both", ls="--", lw=0.4)
        if yticks is not None:
            ax.set_yticks(yticks)

    # Prepare static DataFrames (declared-in-script results) ---------------
    try:
        static_ns_df = result_df
    except NameError:
        static_ns_df = None

    try:
        static_mw3_df = result_mw3_df
    except NameError:
        static_mw3_df = None

    static_rel_df = None
    try:
        static_rel_df = pd.DataFrame([resnet, gcresnet, condnet])
    except NameError:
        static_rel_df = None

    # Draw NS, MW³, Relativistic Maxwell panels ----------------------------
    _draw(axes[0], "ns", ns_ylim, ns_yticks, static_df=static_ns_df)
    mw3_style_map = {
        "Cond. CSCNN (Ours)": {"linestyle": "--", "marker": "s", "linewidth": 2},
        "CSCNN": {"linestyle": "-", "marker": "s", "linewidth": 2},
    }
    _draw(axes[1], "mw3", mw_ylim, mw_yticks, static_df=static_mw3_df, style_map=mw3_style_map)

    # Relativistic Maxwell panel (static only)
    ax_rel = axes[2]
    _draw_static_from_df(ax_rel, static_rel_df, style_map={
        "Cond. CSCNN (Ours)": {"marker": "s", "linestyle": "--"}
    })
    ax_rel.set_yscale("log")
    if rel_ylim is not None:
        ax_rel.set_ylim(rel_ylim)
    ax_rel.set_xlabel(r"No. of Training Simulations")
    ax_rel.grid(True, which="both", ls="--", lw=0.4)
    if rel_yticks is not None:
        ax_rel.set_yticks(rel_yticks)
        ax_rel.yaxis.set_minor_locator(mticker.NullLocator())
        ax_rel.yaxis.set_minor_formatter(mticker.NullFormatter())
    else:
        # Fewer, clean log ticks by default
        ax_rel.yaxis.set_major_locator(mticker.LogLocator(base=10, numticks=6))
        ax_rel.yaxis.set_minor_locator(mticker.NullLocator())
        ax_rel.yaxis.set_minor_formatter(mticker.NullFormatter())

    # Shared Y-label -------------------------------------------------------
    axes[0].set_ylabel("MSE")

    # Legend: aggregate unique labels from all axes ------------------------
    handle_map = {}
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in handle_map:
                handle_map[label] = handle
    handles = list(handle_map.values())
    labels = list(handle_map.keys())
    leg =  fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(labels), columnspacing=0.2, handlelength=0.6, handletextpad=0.2, frameon=True, fancybox=True)
    frame = leg.get_frame()
    frame.set_edgecolor("black")


    fig.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for legend
    fig.savefig("figures/total_mse_side_by_side.pdf",
                bbox_inches="tight",
                pad_inches=0.5)
    plt.show()


if __name__ == "__main__":
    # NS
    #plot_ns_total_mse_vs_numsamples()
    #plot_ns_vector_mse_vs_numsamples()
    #plot_ns_scalar_mse_vs_numsamples()

    # MW³
    #plot_mw3_total_mse_vs_numsamples()
    #plot_mw3_vector_mse_vs_numsamples()
    #plot_mw3_scalar_mse_vs_numsamples()

    # Combined total-MSE comparison
    plot_total_mse_side_by_side()

    # Bar plot of loss differences at the chosen data size
    #bar_plot()

    # Bar plot (percentage improvement)
    #bar_plot_percent()

