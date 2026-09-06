# -*- coding: utf-8 -*-
"""
Aircraft engine supply-chain network:
collapse-point and topology-metric analysis.

Key revisions in this version
-----------------------------
1. Figures are saved as JPG only at 300 dpi.
2. All figure text uses Times New Roman, 12 pt, but ordinary labels are not italic.
3. In Figure 2, only mathematical symbols are italic:
   - x and y in the fitted equation;
   - R and p in the statistics line.
4. Figure 3 keeps the lower-right boxplot inset with jittered observations,
   and the legend is placed in the upper-left corner.
5. Figure 5 compares 2015, 2020 and 2023 only, using seven selected metrics:
   Average Path Length, Global Efficiency, Average In-Degree,
   Average Out-Degree, Spectral Radius, Frobenius Norm,
   and Degree Assortativity.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import stats
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates

warnings.filterwarnings("ignore")

font_size = 19.8

# ============================================================
# 1. Global plotting style
# ============================================================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "figure.titlesize": font_size,
    "axes.unicode_minus": False,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Times New Roman",
    "mathtext.it": "Times New Roman:italic",
    #"mathtext.bf": "Times New Roman:bold"
})

sns.set_theme(style="whitegrid", rc={
    "font.family": "Times New Roman",
    "font.size": font_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size
})

DPI = 300

MACARON = [
    "#79C7C5", "#F4D35E", "#EE964B", "#F6A6B2",
    "#A8DADC", "#B8A1D9", "#9ED2A1", "#FFD6A5"
]


# ============================================================
# 2. File paths
# ============================================================
base_path = Path(r"E:\论文写作\35 TRE\代码")

collapse_file = (
    base_path
    / "5 30年分析结果_去边"
    / "collapse_points_summary.csv"
)

topology_file = (
    base_path
    / "6 网络拓扑结构"
    / "数据"
    / "network_topology_metrics_complete_2006_2024.csv"
)

output_path = (
    base_path
    / "7 崩溃点与网络拓扑结构之间关系结果"
    / "作图"
)
output_path.mkdir(exist_ok=True, parents=True)


# ============================================================
# 3. Utility functions
# ============================================================
def save_jpg(fig, filename):
    """Save one figure as JPG only."""
    fig.savefig(
        output_path / filename,
        dpi=DPI,
        format="jpg",
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


def clean_xy(data, x_col, y_col):
    """Drop missing and non-finite values."""
    temp = data[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    return temp[x_col].to_numpy(dtype=float), temp[y_col].to_numpy(dtype=float)


def linear_fit_with_ci(x, y, confidence=0.95, n_grid=300):
    """OLS linear fit and confidence band for the mean fitted response."""
    if len(x) < 3:
        raise ValueError("At least three valid observations are required.")

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    x_grid = np.linspace(np.min(x), np.max(x), n_grid)
    y_fit = intercept + slope * x_grid

    y_hat = intercept + slope * x
    residuals = y - y_hat
    n = len(x)
    dof = n - 2
    s_err = np.sqrt(np.sum(residuals ** 2) / dof)
    x_mean = np.mean(x)
    sxx = np.sum((x - x_mean) ** 2)

    if sxx == 0:
        ci = np.zeros_like(x_grid)
    else:
        t_critical = stats.t.ppf((1 + confidence) / 2, dof)
        ci = t_critical * s_err * np.sqrt(
            1 / n + (x_grid - x_mean) ** 2 / sxx
        )

    return {
        "slope": slope,
        "intercept": intercept,
        "r2": r_value ** 2,
        "p_value": p_value,
        "std_err": std_err,
        "x_grid": x_grid,
        "y_fit": y_fit,
        "lower": y_fit - ci,
        "upper": y_fit + ci
    }


# ============================================================
# Regression-equation formatting
# ============================================================
def format_equation_mathtext(slope, intercept, x_symbol):
    """
    Format the regression equation using the actual mathematical variables.

    Examples:
        q* = 0.0004 rho(A) + 0.7685
        q* = 0.0002 ||A||_F + 0.7734
        q* = 0.0004 k_bar + 0.7780
    """
    sign = "+" if intercept >= 0 else "-"

    return (
        rf"$q^* = {slope:.4f}{x_symbol} "
        rf"{sign} {abs(intercept):.4f}$"
    )


# ============================================================
# Joint regression panel
# ============================================================
def draw_joint_regression_panel(
    fig,
    outer_spec,
    data,
    x_col,
    y_col,
    x_label,
    y_label,
    panel_label,
    color,
    x_symbol,
    y_lim=(0.800, 0.860),
    y_ticks=None,
    draw_right_marginal=True,
    show_y_label=True,
    show_y_tick_labels=True,
    xticks=None,
    xtick_labels=None
):
    """
    Draw one joint scatter-regression panel with marginal densities,
    OLS fit, 95% confidence interval and an internal panel label.

    x_symbol controls the mathematical variable displayed in
    the regression equation.
    """

    inner = GridSpecFromSubplotSpec(
        2, 2,
        subplot_spec=outer_spec,
        width_ratios=[5.0, 0.6],
        height_ratios=[0.6, 5.0],
        wspace=0.0,
        hspace=0.05
    )

    ax_top = fig.add_subplot(inner[0, 0])
    ax_joint = fig.add_subplot(inner[1, 0], sharex=ax_top)
    ax_right = fig.add_subplot(inner[1, 1], sharey=ax_joint)
    ax_blank = fig.add_subplot(inner[0, 1])
    ax_blank.axis("off")

    # Hide right marginal distribution when it is not required
    if not draw_right_marginal:
        ax_right.set_visible(False)

    # --------------------------------------------------------
    # Clean data and perform linear regression
    # --------------------------------------------------------
    x, y = clean_xy(data, x_col, y_col)
    fit = linear_fit_with_ci(x, y)

    # --------------------------------------------------------
    # Scatter points
    # --------------------------------------------------------
    ax_joint.scatter(
        x,
        y,
        s=34,
        alpha=0.85,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=3
    )

    # --------------------------------------------------------
    # 95% confidence interval
    # --------------------------------------------------------
    ax_joint.fill_between(
        fit["x_grid"],
        fit["lower"],
        fit["upper"],
        color=color,
        alpha=0.22,
        linewidth=0,
        label="95% confidence interval",
        zorder=1
    )

    # --------------------------------------------------------
    # Regression equation
    #
    # The x variable is panel-specific:
    # (a) rho(A)
    # (b) ||A||_F
    # (c) k_bar
    # --------------------------------------------------------
    equation = format_equation_mathtext(
        fit["slope"],
        fit["intercept"],
        x_symbol
    )

    fit_label = (
        f"Linear fit: {equation}\n"
        f"$\\mathit{{R}}^2$ = {fit['r2']:.3f}, "
        f"$\\mathit{{p}}$ = {fit['p_value']:.3f}"
    )

    # --------------------------------------------------------
    # Regression line
    # --------------------------------------------------------
    ax_joint.plot(
        fit["x_grid"],
        fit["y_fit"],
        color="#D65F5F",
        linewidth=1.8,
        label=fit_label,
        zorder=4
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------
    ax_joint.set_xlabel(x_label)

    if xticks is not None:
        ax_joint.set_xticks(xticks)

        if xtick_labels is not None:
            ax_joint.set_xticklabels(xtick_labels)
        else:
            ax_joint.set_xticklabels(
                [str(tick) for tick in xticks]
            )

    # --------------------------------------------------------
    # Y-axis
    # --------------------------------------------------------
    if show_y_label:
        ax_joint.set_ylabel(y_label)
    else:
        ax_joint.set_ylabel("")

    ax_joint.set_ylim(*y_lim)

    if y_ticks is None:
        y_ticks = np.linspace(
            y_lim[0],
            y_lim[1],
            4
        )

    ax_joint.set_yticks(y_ticks)

    if not show_y_tick_labels:
        ax_joint.tick_params(
            axis="y",
            labelleft=False
        )

    # --------------------------------------------------------
    # Grid and legend
    # --------------------------------------------------------
    ax_joint.grid(
        True,
        alpha=0.25
    )

    ax_joint.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.90,
        handlelength=2.0,
        borderpad=0.6,
        fontsize=font_size - 2
    )

    # --------------------------------------------------------
    # Panel label: (a), (b), (c)
    # --------------------------------------------------------
    ax_joint.text(
        0.03,
        0.96,
        panel_label,
        transform=ax_joint.transAxes,
        ha="left",
        va="top",
        fontsize=font_size,
        zorder=10
    )

    # ========================================================
    # Top marginal distribution
    # ========================================================
    try:
        if np.unique(x).size > 1 and np.std(x) > 0:

            kde_x = stats.gaussian_kde(x)

            x_grid = np.linspace(
                np.min(x),
                np.max(x),
                300
            )

            density_x = kde_x(x_grid)

            ax_top.fill_between(
                x_grid,
                0,
                density_x,
                color=color,
                alpha=0.75
            )

            ax_top.plot(
                x_grid,
                density_x,
                color="#555555",
                linewidth=0.9
            )

        else:

            ax_top.hist(
                x,
                bins=min(5, max(1, len(x))),
                density=True,
                color=color,
                alpha=0.75
            )

    except (np.linalg.LinAlgError, ValueError):

        ax_top.hist(
            x,
            bins=min(6, max(1, len(x))),
            density=True,
            color=color,
            alpha=0.75
        )

    ax_top.set_ylabel("")
    ax_top.set_xlabel("")

    ax_top.tick_params(
        axis="x",
        labelbottom=False
    )

    ax_top.tick_params(
        axis="y",
        left=False,
        labelleft=False
    )

    ax_top.grid(False)

    sns.despine(
        ax=ax_top,
        left=True,
        bottom=True
    )

    # ========================================================
    # Right marginal distribution
    # ========================================================
    if draw_right_marginal:

        y_density_values = y[
            (y >= y_lim[0]) &
            (y <= y_lim[1])
        ]

        if y_density_values.size == 0:
            y_density_values = y

        try:

            if (
                np.unique(y_density_values).size > 1
                and np.std(y_density_values) > 0
            ):

                kde_y = stats.gaussian_kde(
                    y_density_values
                )

                y_grid = np.linspace(
                    y_lim[0],
                    y_lim[1],
                    300
                )

                density_y = kde_y(y_grid)

                ax_right.fill_betweenx(
                    y_grid,
                    0,
                    density_y,
                    color=color,
                    alpha=0.75
                )

                ax_right.plot(
                    density_y,
                    y_grid,
                    color="#555555",
                    linewidth=0.9
                )

            else:

                ax_right.hist(
                    y_density_values,
                    bins=min(
                        5,
                        max(1, len(y_density_values))
                    ),
                    density=True,
                    orientation="horizontal",
                    color=color,
                    alpha=0.75
                )

        except (np.linalg.LinAlgError, ValueError):

            ax_right.hist(
                y_density_values,
                bins=min(
                    6,
                    max(1, len(y_density_values))
                ),
                density=True,
                orientation="horizontal",
                color=color,
                alpha=0.75
            )

        ax_right.set_ylim(*y_lim)
        ax_right.set_yticks(y_ticks)

        ax_right.set_xlabel("")
        ax_right.set_ylabel("")

        ax_right.tick_params(
            axis="x",
            bottom=False,
            labelbottom=False
        )

        ax_right.tick_params(
            axis="y",
            left=False,
            labelleft=False
        )

        ax_right.grid(False)

        sns.despine(
            ax=ax_right,
            left=True,
            bottom=True
        )

    else:

        ax_right.set_visible(False)


def add_boxplot_inset(ax, values, color):
    """Add a lower-right boxplot inset with jittered observations."""
    inset = ax.inset_axes([0.79, 0.09, 0.17, 0.34])
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]

    if clean_values.size == 0:
        inset.axis("off")
        return

    bp = inset.boxplot(
        clean_values,
        vert=True,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#555555", "linewidth": 1.1},
        whiskerprops={"color": "#666666", "linewidth": 0.9},
        capprops={"color": "#666666", "linewidth": 0.9}
    )

    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.58)
    bp["boxes"][0].set_edgecolor("#666666")

    rng = np.random.default_rng(2026)
    jitter_x = 1.0 + rng.uniform(-0.10, 0.10, size=len(clean_values))
    inset.scatter(
        jitter_x,
        clean_values,
        s=10,
        color=color,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.82,
        zorder=3
    )

    inset.set_xticks([])
    inset.set_yticks([])
    inset.tick_params(
        axis="both",
        which="both",
        bottom=False,
        left=False,
        labelbottom=False,
        labelleft=False
    )
    inset.grid(False)


def resolve_column(dataframe, candidates, display_name):
    """Return the first existing column from a list of candidate names."""
    for column in candidates:
        if column in dataframe.columns:
            return column
    raise KeyError(
        f"Cannot find a column for '{display_name}'. "
        f"Tried: {candidates}. "
        f"Available columns are: {list(dataframe.columns)}"
    )


def nice_axis_limits_and_ticks(values, n_ticks=5, include_zero=False):
    """Compute readable panel-specific y-axis limits and ticks."""
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]

    if clean_values.size == 0:
        return (0.0, 1.0), np.linspace(0.0, 1.0, n_ticks)

    value_min = float(np.min(clean_values))
    value_max = float(np.max(clean_values))

    if include_zero and value_min > 0:
        value_min = 0.0

    if np.isclose(value_min, value_max):
        margin = max(abs(value_min) * 0.05, 1.0)
    else:
        margin = (value_max - value_min) * 0.08

    lower = value_min - margin
    upper = value_max + margin

    if include_zero and lower < 0 <= value_min:
        lower = 0.0

    ticks = np.linspace(lower, upper, n_ticks)
    return (lower, upper), ticks


# ============================================================
# 4. Read and merge data
# ============================================================
print("Reading data...")
collapse_df = pd.read_csv(collapse_file)
topology_df = pd.read_csv(topology_file)

collapse_df = collapse_df.rename(columns={"Year": "year"})
topology_df = topology_df.rename(columns={"Year": "year"})

merged_all_df = pd.merge(collapse_df, topology_df, on="year", how="inner")
merged_all_df = merged_all_df.sort_values("year").reset_index(drop=True)

# Preserve the previous rule for all analyses except the radar chart.
merged_df = merged_all_df[merged_all_df["year"] != 2024].copy()
merged_df = merged_df.sort_values("year").reset_index(drop=True)

for _df in (merged_df, merged_all_df):
    _df["avg_degree"] = _df["avg_in_degree"] + _df["avg_out_degree"]

node_count_source = resolve_column(
    merged_all_df,
    [
        "num_nodes", "node_count", "number_of_nodes", "n_nodes",
        "nodes", "Node_Count", "Number_of_Nodes"
    ],
    "Network Node Count"
)

edge_count_source = resolve_column(
    merged_all_df,
    [
        "num_edges", "edge_count", "number_of_edges", "n_edges",
        "edges", "Edge_Count", "Number_of_Edges"
    ],
    "Network Edge Count"
)

for _df in (merged_df, merged_all_df):
    _df["network_node_count"] = pd.to_numeric(
        _df[node_count_source], errors="coerce"
    )
    _df["network_edge_count"] = pd.to_numeric(
        _df[edge_count_source], errors="coerce"
    )

    # Total trade value is calculated according to the requested rule:
    # total trade value = average degree x number of network nodes.
    # Here, average degree is defined as average in-degree plus
    # average out-degree.
    _df["total_trade_value"] = (
        _df["avg_degree"] * _df["network_node_count"]
    )

print(f"Data shape after merging and removing 2024: {merged_df.shape}")
print(f"Year range: {merged_df['year'].min()} - {merged_df['year'].max()}")


# ============================================================
# 5. Metric dictionaries
# ============================================================
original_topology_metrics = {
    "avg_clustering": "Average Clustering coefficient",
    "avg_path_length": "Average path length",
    "diameter": "Network diameter",
    "efficiency_global": "Global efficiency",
    "avg_in_degree": "Average in-degree",
    "avg_out_degree": "Average out-degree",
    "density": "Network density"
}

new_matrix_metrics = {
    "matrix_adj_spectral_radius": "Spectral radius",
    "matrix_adj_frobenius_norm": "Frobenius norm",
    "matrix_adj_sparsity": "Network sparsity",
    "matrix_degree_assortativity": "Degree assortativity",
    "matrix_lap_fiedler_value": "Fiedler eigenvalue"
}

topology_metrics = {**original_topology_metrics, **new_matrix_metrics}
collapse_col = "Collapse_q_total"


# ============================================================
# 6. Correlation analysis
# ============================================================
correlation_results = []
for metric_key, metric_name in topology_metrics.items():
    x, y = clean_xy(merged_df, metric_key, collapse_col)
    if len(x) >= 3 and np.std(x) > 0 and np.std(y) > 0:
        corr_coef, p_value = stats.pearsonr(x, y)
    else:
        corr_coef, p_value = np.nan, np.nan

    correlation_results.append({
        "Metric": metric_name,
        "Metric_Code": metric_key,
        "Metric_Type": "Original" if metric_key in original_topology_metrics else "Matrix",
        "Correlation_Coefficient": corr_coef,
        "P_Value": p_value,
        "Significant": bool(p_value < 0.05) if np.isfinite(p_value) else False
    })

correlation_df = pd.DataFrame(correlation_results).sort_values(
    "Correlation_Coefficient", ascending=False
)
correlation_df.to_csv(output_path / "correlation_analysis.csv", index=False)

all_vars = [collapse_col] + list(topology_metrics.keys())
all_var_names = ["Collapse point"] + list(topology_metrics.values())
full_corr_matrix = merged_df[all_vars].corr(method="pearson")
full_corr_matrix.columns = all_var_names
full_corr_matrix.index = all_var_names
full_corr_matrix.to_csv(output_path / "full_correlation_matrix.csv")


'''
# ============================================================
# 7. Figure 1a and 1b: scatter matrices
# ============================================================
data_original = merged_df[[collapse_col] + list(original_topology_metrics.keys())].copy()
data_original.columns = ["Collapse Point", *original_topology_metrics.values()]
axes = pd.plotting.scatter_matrix(
    data_original, alpha=0.75, figsize=(14, 14), diagonal="hist",
    hist_kwds={"bins": 10, "alpha": 0.75, "color": MACARON[0]}
)
fig = axes[0, 0].get_figure()
fig.suptitle("Relationship Matrix: Collapse Point and Original Topology Metrics", y=0.995)
fig.tight_layout()
save_jpg(fig, "1a_scatter_matrix_original.jpg")

data_new = merged_df[[collapse_col] + list(new_matrix_metrics.keys())].copy()
data_new.columns = ["Collapse Point", *new_matrix_metrics.values()]
axes = pd.plotting.scatter_matrix(
    data_new, alpha=0.75, figsize=(12, 12), diagonal="hist",
    hist_kwds={"bins": 10, "alpha": 0.75, "color": MACARON[2]}
)
fig = axes[0, 0].get_figure()
fig.suptitle("Relationship Matrix: Collapse Point and Matrix-Based Metrics", y=0.995)
fig.tight_layout()
save_jpg(fig, "1b_scatter_matrix_new.jpg")
'''


# ============================================================
# Figure 2:
# Collapse threshold q* vs three selected network metrics
# ============================================================

# ------------------------------------------------------------
# Selected metrics and their mathematical x-axis labels
# ------------------------------------------------------------
selected_scatter_metrics = [
    (
        "matrix_adj_spectral_radius",
        r"Spectral radius $\rho(A)$",
        MACARON[1]
    ),
    (
        "matrix_adj_frobenius_norm",
        r"Frobenius norm $\Vert A \Vert_F$",
        MACARON[2]
    ),
    (
        "avg_degree",
        r"Average degree $\bar{k}$",
        MACARON[0]
    )
]


# ------------------------------------------------------------
# Mathematical symbols used in the regression equations
# ------------------------------------------------------------
regression_x_symbols = [
    r"\rho(A)",
    r"\Vert A \Vert_F",
    r"\bar{k}"
]


# ------------------------------------------------------------
# Figure layout
# ------------------------------------------------------------
fig = plt.figure(
    figsize=(18, 6)
)

outer = GridSpec(
    1,
    3,
    figure=fig,
    wspace=-0.03
)

panel_letters = [
    "(a)",
    "(b)",
    "(c)"
]

collapse_y_lim = (
    0.79,
    0.85
)

collapse_y_ticks = np.linspace(
    0.79,
    0.85,
    4
)


# ============================================================
# (a) Spectral radius rho(A)
# ============================================================
draw_joint_regression_panel(
    fig=fig,
    outer_spec=outer[0, 0],
    data=merged_df,

    x_col=selected_scatter_metrics[0][0],
    y_col=collapse_col,

    x_label=selected_scatter_metrics[0][1],
    y_label=r"Collapse threshold $q^*$",

    panel_label=panel_letters[0],
    color=selected_scatter_metrics[0][2],

    # Regression equation:
    # q* = beta1 rho(A) + beta0
    x_symbol=regression_x_symbols[0],

    y_lim=collapse_y_lim,
    y_ticks=collapse_y_ticks,

    draw_right_marginal=False,

    show_y_label=True,
    show_y_tick_labels=True,

    xticks=[
        120,
        140,
        160
    ]
)


# ============================================================
# (b) Frobenius norm ||A||_F
# ============================================================
draw_joint_regression_panel(
    fig=fig,
    outer_spec=outer[0, 1],
    data=merged_df,

    x_col=selected_scatter_metrics[1][0],
    y_col=collapse_col,

    x_label=selected_scatter_metrics[1][1],
    y_label=r"Collapse threshold $q^*$",

    panel_label=panel_letters[1],
    color=selected_scatter_metrics[1][2],

    # Regression equation:
    # q* = beta1 ||A||_F + beta0
    x_symbol=regression_x_symbols[1],

    y_lim=collapse_y_lim,
    y_ticks=collapse_y_ticks,

    draw_right_marginal=False,

    show_y_label=False,
    show_y_tick_labels=False,

    xticks=[
        270,
        310,
        350,
        390
    ]
)


# ============================================================
# (c) Average degree k-bar
# ============================================================
draw_joint_regression_panel(
    fig=fig,
    outer_spec=outer[0, 2],
    data=merged_df,

    x_col=selected_scatter_metrics[2][0],
    y_col=collapse_col,

    x_label=selected_scatter_metrics[2][1],
    y_label=r"Collapse threshold $q^*$",

    panel_label=panel_letters[2],
    color=selected_scatter_metrics[2][2],

    # Regression equation:
    # q* = beta1 k_bar + beta0
    x_symbol=regression_x_symbols[2],

    y_lim=collapse_y_lim,
    y_ticks=collapse_y_ticks,

    draw_right_marginal=True,

    show_y_label=False,
    show_y_tick_labels=False,

    xticks=[
        120,
        140,
        160
    ]
)


# ============================================================
# Save Figure 2
# ============================================================
save_jpg(
    fig,
    "2_collapse_vs_all_metrics_scatter.jpg"
)


# ============================================================
# 9. Figure 3:
# six selected time-series panels, 2 x 3
# ============================================================
time_series_metrics = [
    ("avg_degree", "Average Degree"),
    ("matrix_adj_spectral_radius", "Spectral radius"),
    ("matrix_adj_frobenius_norm", "Frobenius norm"),
    ("network_node_count", "Number of nodes"),
    ("network_edge_count", "Number of edges"),
    ("total_trade_value", "Total trade value")
]

# ------------------------------------------------------------
# Each subplot has its own independently coded y-axis settings.
# Edit the ylim and yticks for each metric separately as needed.
# ------------------------------------------------------------

avg_degree_ylim = (
    #float(np.nanmin(merged_df["avg_degree"])) * 0.95,
    #float(np.nanmax(merged_df["avg_degree"])) * 1.05
    90,
    178
)
avg_degree_yticks = np.linspace(
    avg_degree_ylim[0],
    avg_degree_ylim[1],
    5
)

spectral_radius_ylim = (
    110,
    180
)
spectral_radius_yticks = np.linspace(
    spectral_radius_ylim[0],
    spectral_radius_ylim[1],
    5
)

frobenius_norm_ylim = (
    250,
    430
)
frobenius_norm_yticks = np.linspace(
    frobenius_norm_ylim[0],
    frobenius_norm_ylim[1],
    5
)

node_count_ylim = (
    100,
    188
)

node_count_yticks = np.linspace(
    node_count_ylim[0],
    node_count_ylim[1],
    5
)

edge_count_ylim = (
    500,
    1200
)

edge_count_yticks = np.linspace(
    edge_count_ylim[0],
    edge_count_ylim[1],
    5
)

total_trade_value_ylim = (
    10000,
    28000
)

total_trade_value_yticks = np.linspace(
    total_trade_value_ylim[0],
    total_trade_value_ylim[1],
    5
)

time_series_axis_settings = {
    "avg_degree": {
        "ylim": avg_degree_ylim,
        "yticks": avg_degree_yticks
    },
    "matrix_adj_spectral_radius": {
        "ylim": spectral_radius_ylim,
        "yticks": spectral_radius_yticks
    },
    "matrix_adj_frobenius_norm": {
        "ylim": frobenius_norm_ylim,
        "yticks": frobenius_norm_yticks
    },
    "network_node_count": {
        "ylim": node_count_ylim,
        "yticks": node_count_yticks
    },
    "network_edge_count": {
        "ylim": edge_count_ylim,
        "yticks": edge_count_yticks
    },
    "total_trade_value": {
        "ylim": total_trade_value_ylim,
        "yticks": total_trade_value_yticks
    }
}

fig, axes = plt.subplots(
    2, 3,
    figsize=(15, 8.5),
    constrained_layout=True
)
axes = axes.flatten()

for i, ((metric_key, metric_name), ax) in enumerate(
    zip(time_series_metrics, axes)
):
    color = MACARON[i]
    values = merged_df[metric_key].to_numpy(dtype=float)
    years = merged_df["year"].to_numpy()

    ax.plot(
        years,
        values,
        marker="o",
        markersize=4.5,
        linewidth=1.5,
        color=color,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.0,
        zorder=5
    )

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        ax.set_visible(False)
        continue

    value_min = np.nanmin(finite_values)
    value_max = np.nanmax(finite_values)

    ax.axhline(
        value_min,
        color=color,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=f"Minimum = {value_min:.0f}",
        zorder=4
    )
    ax.axhline(
        value_max,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=f"Maximum = {value_max:.0f}",
        zorder=4
    )

    axis_setting = time_series_axis_settings[metric_key]
    ax.set_ylim(*axis_setting["ylim"])
    ax.set_yticks(axis_setting["yticks"])

    ax.set_xlabel("Year")
    ax.set_ylabel(metric_name)
    ax.grid(True, alpha=0.25)
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 1.03),
        fontsize=font_size - 3,
        frameon=True,
        framealpha=0.65,
        labelspacing=0.2
    )
    
    legend.set_zorder(2)

    ax.text(
        0.97,
        0.95,
        f"({chr(97 + i)})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=font_size-2,
        #fontweight="bold"
    )

    add_boxplot_inset(ax, finite_values, color)

save_jpg(fig, "3_time_series_all_metrics.jpg")


# ============================================================
# 10. Figure 4: correlation heatmap
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(full_corr_matrix, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    full_corr_matrix,
    mask=mask,
    cmap=cmap,
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=0.8,
    linecolor="white",
    annot=True,
    fmt=".2f",
    annot_kws={"size": 9, "family": "Times New Roman"},
    cbar_kws={"shrink": 0.80, "label": "Correlation Coefficient"},
    ax=ax
)

#ax.set_title("Correlation Heatmap of Network Metrics and Collapse point", pad=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
save_jpg(fig, "4_correlation_heatmap_all.jpg")


'''
# ============================================================
# 11. Figure 5: radar chart, 2015/2020/2023 and 7 metrics only
# ============================================================
radar_metrics = {
    "avg_path_length": "Average path length",
    "efficiency_global": "Global efficiency",
    "avg_in_degree": "Average in-degree",
    "avg_out_degree": "Average out-degree",
    "matrix_adj_spectral_radius": "Spectral radius",
    "matrix_adj_frobenius_norm": "Frobenius norm",
    "matrix_degree_assortativity": "Degree assortativity"
}

required_years = [2015, 2020, 2023]
available_years = set(merged_all_df["year"].astype(int).tolist())
missing_years = [year for year in required_years if year not in available_years]
if missing_years:
    raise ValueError(
        "The radar chart requires data for 2015, 2020 and 2023. "
        f"Missing year(s): {missing_years}"
    )

# Standardize the selected metrics using all available years so that the
# three selected years are compared on a common scale.
radar_scaler = StandardScaler()
radar_scaled = radar_scaler.fit_transform(
    merged_all_df[list(radar_metrics.keys())]
)
radar_scaled_df = pd.DataFrame(
    radar_scaled,
    columns=list(radar_metrics.keys())
)
radar_scaled_df["year"] = merged_all_df["year"].values

categories = list(radar_metrics.values())
n_categories = len(categories)
angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
for i, year in enumerate(required_years):
    row = radar_scaled_df[radar_scaled_df["year"] == year]
    values = row[list(radar_metrics.keys())].iloc[0].tolist()
    values += values[:1]

    color = MACARON[i]
    ax.plot(
        angles, values,
        marker="o", linewidth=1.7, markersize=4.5,
        label=f"Year {year}", color=color
    )
    ax.fill(angles, values, color=color, alpha=0.12)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10)
ax.set_title("Selected Network Topology Metrics Radar Chart", pad=18)
ax.legend(loc="upper right", bbox_to_anchor=(1.24, 1.10))
ax.grid(True, alpha=0.30)
fig.tight_layout()
save_jpg(fig, "5_radar_chart_all_metrics.jpg")
'''


'''
# ============================================================
# 12. Figure 6: parallel coordinates
# ============================================================
scaler = StandardScaler()
metrics_scaled = scaler.fit_transform(merged_df[list(topology_metrics.keys())])
metrics_scaled_df = pd.DataFrame(metrics_scaled, columns=list(topology_metrics.keys()))
metrics_scaled_df["year"] = merged_df["year"].values

plot_data = metrics_scaled_df.copy()
plot_data["year"] = plot_data["year"].astype(str)
fig, ax = plt.subplots(figsize=(15, 8))
parallel_coordinates(
    plot_data,
    "year",
    cols=list(topology_metrics.keys()),
    color=[MACARON[i % len(MACARON)] for i in range(plot_data["year"].nunique())],
    linewidth=1.2,
    alpha=0.70,
    ax=ax
)
ax.set_xlabel("Network topology metrics")
ax.set_ylabel("Standardized Value")
ax.set_title("Parallel Coordinates Plot of Network Topology Metrics")
ax.legend(title="Year", bbox_to_anchor=(1.02, 1), loc="upper left",
          fontsize=8, title_fontsize=9)
ax.grid(True, alpha=0.25)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
save_jpg(fig, "6_parallel_coordinates_all.jpg")
'''


'''
# ============================================================
# 13. Figure 7: boxplots for all metrics
# ============================================================
all_box_metrics = [(collapse_col, "Collapse point"), *list(topology_metrics.items())]
fig, axes = plt.subplots(4, 4, figsize=(14, 12), constrained_layout=True)
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i >= len(all_box_metrics):
        ax.set_visible(False)
        continue

    metric_key, metric_name = all_box_metrics[i]
    values = merged_df[metric_key].dropna().values
    color = MACARON[i % len(MACARON)]
    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#555555", "linewidth": 1.2}
    )
    bp["boxes"][0].set_facecolor(color)
    bp["boxes"][0].set_alpha(0.70)
    bp["boxes"][0].set_edgecolor("#666666")
    ax.set_xticks([1])
    ax.set_xticklabels([metric_name], rotation=20, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(f"({chr(97 + i)}) {metric_name}")
    ax.grid(True, axis="y", alpha=0.25)

save_jpg(fig, "7_boxplots_all_metrics.jpg")
'''


# ============================================================
# 14. Statistical summary CSV and table
# ============================================================
summary_cols = [collapse_col] + list(topology_metrics.keys())
stats_summary = merged_df[summary_cols].describe().round(4)
corr_map = correlation_df.set_index("Metric_Code")
corr_row = [np.nan]
p_row = [np.nan]

for metric_key in topology_metrics.keys():
    corr_row.append(corr_map.loc[metric_key, "Correlation_Coefficient"])
    p_row.append(corr_map.loc[metric_key, "P_Value"])

stats_summary.loc["Correlation"] = corr_row
stats_summary.loc["P_Value"] = p_row
stats_summary.to_csv(output_path / "8_statistical_summary_all_metrics.csv")

fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")
table = ax.table(
    cellText=stats_summary.round(4).values,
    rowLabels=stats_summary.index,
    colLabels=["Collapse point", *topology_metrics.values()],
    cellLoc="center",
    loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.0, 1.35)
for col_idx in range(len(stats_summary.columns)):
    header = table[(0, col_idx)]
    header.set_facecolor(MACARON[0])
    header.set_text_props(weight="bold", color="black")
ax.set_title("Statistical Summary of Network Metrics and Collapse point", pad=10)
fig.tight_layout()
save_jpg(fig, "8_statistical_summary_table_all.jpg")


# ============================================================
# 15. Figure 9: quadratic fits for all 12 metrics
# ============================================================
fig, axes = plt.subplots(4, 3, figsize=(14, 15), constrained_layout=True)
axes = axes.flatten()

for i, (metric_key, metric_name) in enumerate(topology_metrics.items()):
    ax = axes[i]
    x, y = clean_xy(merged_df, metric_key, collapse_col)
    color = MACARON[i % len(MACARON)]
    ax.scatter(x, y, s=30, alpha=0.75, color=color,
               edgecolor="white", linewidth=0.6, label="Observed data")

    if len(x) >= 3 and np.unique(x).size >= 3:
        coeff = np.polyfit(x, y, 2)
        poly = np.poly1d(coeff)
        x_smooth = np.linspace(x.min(), x.max(), 200)
        y_smooth = poly(x_smooth)
        y_pred = poly(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        ax.plot(
            x_smooth, y_smooth,
            color="#D65F5F", linewidth=1.6,
            label=f"Quadratic fit, $\\mathit{{R}}^2$ = {r2:.3f}"
        )

    ax.set_xlabel(metric_name)
    ax.set_ylabel("Collapse point")
    ax.set_title(f"({chr(97 + i)}) {metric_name}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

save_jpg(fig, "9_polynomial_fit_all_metrics.jpg")


'''
# ============================================================
# 16. Figure 10: bubble charts
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
axes = axes.flatten()

sc = axes[0].scatter(
    merged_df["avg_clustering"], merged_df["matrix_adj_spectral_radius"],
    s=merged_df[collapse_col] * 80, c=merged_df["year"], cmap="viridis",
    alpha=0.75, edgecolors="white", linewidth=0.7
)
axes[0].set_xlabel("Average Clustering Coefficient")
axes[0].set_ylabel("Spectral Radius")
axes[0].set_title("(a) Bubble Size: Collapse Point; Color: Year")
axes[0].grid(True, alpha=0.25)
fig.colorbar(sc, ax=axes[0]).set_label("Year")

sc = axes[1].scatter(
    merged_df["matrix_lap_fiedler_value"], merged_df["matrix_adj_frobenius_norm"],
    s=merged_df[collapse_col] * 80, c=merged_df["avg_in_degree"], cmap="plasma",
    alpha=0.75, edgecolors="white", linewidth=0.7
)
axes[1].set_xlabel("Fiedler Eigenvalue")
axes[1].set_ylabel("Frobenius Norm")
axes[1].set_title("(b) Bubble Size: Collapse Point; Color: Average In-Degree")
axes[1].grid(True, alpha=0.25)
fig.colorbar(sc, ax=axes[1]).set_label("Average In-Degree")

sc = axes[2].scatter(
    merged_df["matrix_degree_assortativity"], merged_df["matrix_adj_sparsity"],
    s=merged_df[collapse_col] * 80, c=merged_df["density"], cmap="coolwarm",
    alpha=0.75, edgecolors="white", linewidth=0.7
)
axes[2].set_xlabel("Degree Assortativity")
axes[2].set_ylabel("Network Sparsity")
axes[2].set_title("(c) Bubble Size: Collapse Point; Color: Network Density")
axes[2].grid(True, alpha=0.25)
fig.colorbar(sc, ax=axes[2]).set_label("Network Density")

sc = axes[3].scatter(
    merged_df["year"], merged_df[collapse_col],
    s=merged_df[collapse_col] * 80,
    c=merged_df["matrix_adj_spectral_radius"], cmap="RdYlGn",
    alpha=0.80, edgecolors="black", linewidth=0.6
)
axes[3].set_xlabel("Year")
axes[3].set_ylabel("Collapse Point")
axes[3].set_title("(d) Collapse Point Over Time; Color: Spectral Radius")
axes[3].grid(True, alpha=0.25)
fig.colorbar(sc, ax=axes[3]).set_label("Spectral Radius")

save_jpg(fig, "10_bubble_charts_new_metrics.jpg")
'''


# ============================================================
# 17. Console output
# ============================================================
print("\nAll analyses are complete.")
print(f"Output directory: {output_path}")
print("\nGenerated files:")
for file in sorted(output_path.glob("*.*")):
    print(f"  - {file.name}")

print("\nCorrelation analysis:")
print(correlation_df[[
    "Metric", "Metric_Type", "Correlation_Coefficient", "P_Value", "Significant"
]].to_string(index=False))

print("\nBasic information:")
print(f"Year range: {merged_df['year'].min()} - {merged_df['year'].max()}")
print(f"Number of years: {merged_df['year'].nunique()}")
print(
    f"Collapse-point range: {merged_df[collapse_col].min():.4f} - "
    f"{merged_df[collapse_col].max():.4f}"
)
print(
    f"Collapse-point mean ± SD: {merged_df[collapse_col].mean():.4f} ± "
    f"{merged_df[collapse_col].std():.4f}"
)
