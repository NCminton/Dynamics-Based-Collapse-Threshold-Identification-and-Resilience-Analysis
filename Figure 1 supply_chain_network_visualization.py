# -*- coding: utf-8 -*-
"""
供应链复杂网络可视化 + 度分布分析（最终修改版）
======================================================

本版修改：
1. 全局字体调整为：
   - font.size = 20
   - axes.labelsize = 18
   - axes.titlesize = 18
   - xtick.labelsize = 18
   - ytick.labelsize = 18
   - legend.fontsize = 18

2. 度分布主图横坐标上限改为 40。

3. 双对数坐标系拟合时：
   - 不使用第 1 个数据点；
   - 不使用最后 5 个数据点；
   - 仅使用中间数据点进行线性拟合。

   即对按 degree 从小到大排列后的有效点：
       k_fit = k[1:-5]
       P_fit = P(k)[1:-5]

4. 双对数子图中的拟合信息三行文本增大行间距；字体缩小并置于底层，避免遮挡散点。

5. 网络图节点标签字体由 18 调整为 15，减轻视觉拥挤。

6. 保留上一版全部改进：
   - 数据路径正确；
   - 去除自环；
   - Top 50 weighted degree 网络图；
   - 改进的紧凑网络布局；
   - 不保存 PDF；
   - 不显示图标题；
   - 度分布外层为柱状图；
   - 内嵌双对数图较大；
   - 不使用 tight_layout()，避免 inset_axes 布局警告；
   - Times New Roman；
   - 数学变量使用斜体。

依赖：
    pip install pandas numpy matplotlib networkx scipy
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import linregress
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# 1. 参数设置
# ============================================================

DATA_DIR = Path(r"E:\论文写作\35 TRE\2 每年原始数据\3 网络数据_新")
OUTPUT_DIR = Path(r"E:\论文写作\35 TRE\代码\8 网络图和度分布图")

YEARS = [2023, 2018, 2013, 2008]

TOP_N = 50
LABEL_TOP_N = 30

DIRECTED = True
DEGREE_MODE = "total"   # "total" / "in" / "out"

DPI = 600

# 度分布主图横坐标显示到 40
MAIN_XMAX = 40

# 双对数拟合时排除：
# 第 1 个点 + 最后 5 个点
FIT_EXCLUDE_FIRST = 1
FIT_EXCLUDE_LAST = 5

# ---------- 网络布局参数 ----------
LAYOUT_SEED = 2026
SPRING_ITER = 1500

# 越小越紧凑
SPRING_K = 0.42

# 碰撞分离参数
COLLISION_STEPS = 120
COLLISION_THRESHOLD = 0.085
COLLISION_STRENGTH = 0.006

# 径向压缩参数
RADIAL_COMPRESSION = 0.32

# 马卡龙配色
MACARON_COLORS = [
    "#A8DADC",
    "#F6BD60",
    "#F7A8B8",
    "#B8C0FF",
    "#CDEAC0",
    "#FFD6A5",
    "#CDB4DB",
    "#A9DEF9",
    "#E4C1F9",
    "#FFCAD4",
    "#BDE0FE",
    "#D8E2DC",
]

# ============================================================
# 全局字体
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 24,
    "axes.labelsize": 24,
    "axes.titlesize": 24,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "legend.fontsize": 24,
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
})


# ============================================================
# 2. 数据读取与网络构建
# ============================================================

def read_network_csv(file_path: Path) -> pd.DataFrame:
    """读取并清洗 source / target / weight。"""

    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件：{file_path}")

    df = pd.read_csv(file_path)

    # 统一列名
    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    required = {
        "source",
        "target",
        "weight"
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{file_path.name} 缺少必要列：{sorted(missing)}；"
            f"当前列为：{list(df.columns)}"
        )

    df = df[
        ["source", "target", "weight"]
    ].copy()

    df["source"] = (
        df["source"]
        .astype(str)
        .str.strip()
    )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
    )

    df["weight"] = pd.to_numeric(
        df["weight"],
        errors="coerce"
    )

    # 删除无效值
    df = df.dropna(
        subset=[
            "source",
            "target",
            "weight"
        ]
    )

    df = df[
        (df["source"] != "")
        &
        (df["target"] != "")
    ]

    df = df[
        np.isfinite(df["weight"])
    ]

    df = df[
        df["weight"] > 0
    ]

    # 去除自环
    df = df[
        df["source"] != df["target"]
    ].copy()

    # 合并重复边
    df = (
        df.groupby(
            ["source", "target"],
            as_index=False,
            sort=False
        )["weight"]
        .sum()
    )

    return df


def build_graph(df: pd.DataFrame):
    """构建加权网络，并确保不存在自环。"""

    G = (
        nx.DiGraph()
        if DIRECTED
        else nx.Graph()
    )

    for row in df.itertuples(
        index=False
    ):

        if row.source == row.target:
            continue

        G.add_edge(
            row.source,
            row.target,
            weight=float(row.weight)
        )

    # 再次去除自环
    G.remove_edges_from(
        list(
            nx.selfloop_edges(G)
        )
    )

    return G


# ============================================================
# 3. 网络图：Top 50 weighted degree
# ============================================================

def weighted_total_degree(G):
    """
    计算加权总度 strength。

    有向网络：
        s_i = s_i^in + s_i^out
    """

    if G.is_directed():

        in_s = dict(
            G.in_degree(
                weight="weight"
            )
        )

        out_s = dict(
            G.out_degree(
                weight="weight"
            )
        )

        return {
            n:
            float(
                in_s.get(n, 0.0)
                +
                out_s.get(n, 0.0)
            )
            for n in G.nodes()
        }

    return {
        n: float(v)
        for n, v in G.degree(
            weight="weight"
        )
    }


def robust_minmax(
    values,
    low=0.0,
    high=1.0
):
    """
    log1p 压缩后进行稳健归一化。
    """

    arr = np.asarray(
        values,
        dtype=float
    )

    if len(arr) == 0:
        return np.array([])

    arr = np.log1p(
        np.maximum(
            arr,
            0
        )
    )

    q_low, q_high = np.percentile(
        arr,
        [5, 95]
    )

    if np.isclose(
        q_low,
        q_high
    ):
        q_low, q_high = (
            arr.min(),
            arr.max()
        )

    if np.isclose(
        q_low,
        q_high
    ):
        return np.full_like(
            arr,
            (low + high) / 2.0
        )

    arr = np.clip(
        arr,
        q_low,
        q_high
    )

    x = (
        (arr - q_low)
        /
        (q_high - q_low)
    )

    return (
        low
        +
        x * (high - low)
    )


def get_communities(H):
    """
    社区划分，用于马卡龙配色。
    """

    if H.number_of_nodes() == 0:
        return {}

    U = H.to_undirected()

    try:

        communities = list(
            nx.community.greedy_modularity_communities(
                U,
                weight="weight"
            )
        )

    except Exception:

        communities = [
            set(
                U.nodes()
            )
        ]

    node_to_comm = {}

    for cid, community in enumerate(
        communities
    ):

        for node in community:
            node_to_comm[node] = cid

    return node_to_comm


def separate_close_nodes(
    pos,
    threshold=0.085,
    strength=0.006,
    steps=120
):
    """
    对距离过近的节点施加轻微排斥，
    减少节点真正重叠，但不过度分散网络。
    """

    nodes = list(
        pos.keys()
    )

    coords = np.array(
        [
            pos[n]
            for n in nodes
        ],
        dtype=float
    )

    rng = np.random.default_rng(
        2026
    )

    for _ in range(
        steps
    ):

        disp = np.zeros_like(
            coords
        )

        for i in range(
            len(nodes)
        ):

            for j in range(
                i + 1,
                len(nodes)
            ):

                delta = (
                    coords[i]
                    -
                    coords[j]
                )

                dist = np.linalg.norm(
                    delta
                )

                if dist < 1e-10:

                    delta = rng.uniform(
                        -1e-3,
                        1e-3,
                        size=2
                    )

                    dist = np.linalg.norm(
                        delta
                    )

                if dist < threshold:

                    direction = (
                        delta
                        /
                        dist
                    )

                    push = (
                        strength
                        *
                        (threshold - dist)
                        /
                        threshold
                    )

                    disp[i] += (
                        direction
                        *
                        push
                    )

                    disp[j] -= (
                        direction
                        *
                        push
                    )

        coords += disp

    return {
        n: coords[i]
        for i, n in enumerate(
            nodes
        )
    }


def radial_compress_positions(
    pos,
    compression=0.32
):
    """
    将外围节点平滑向中心压缩，
    使网络整体更加紧凑。

    r_new = r / (1 + compression * r)
    """

    nodes = list(
        pos.keys()
    )

    coords = np.array(
        [
            pos[n]
            for n in nodes
        ],
        dtype=float
    )

    center = coords.mean(
        axis=0
    )

    vec = (
        coords
        -
        center
    )

    r = np.linalg.norm(
        vec,
        axis=1
    )

    scale = np.ones_like(
        r
    )

    nonzero = (
        r > 1e-12
    )

    scale[nonzero] = (
        1.0
        /
        (
            1.0
            +
            compression
            *
            r[nonzero]
        )
    )

    vec = (
        vec
        *
        scale[:, None]
    )

    coords_new = (
        center
        +
        vec
    )

    # 居中
    coords_new -= coords_new.mean(
        axis=0
    )

    # 缩放到统一范围
    max_abs = np.abs(
        coords_new
    ).max()

    if max_abs > 0:
        coords_new = (
            coords_new
            /
            max_abs
        )

    return {
        n: coords_new[i]
        for i, n in enumerate(
            nodes
        )
    }


def compute_better_layout(
    H,
    year
):
    """
    网络布局：
    1. Kamada-Kawai 初始布局；
    2. Spring layout 二次优化；
    3. 轻度碰撞分离；
    4. 径向压缩外围节点。
    """

    U = H.to_undirected().copy()

    # 对布局权重做 log 压缩
    for _, _, data in U.edges(
        data=True
    ):

        data["layout_weight"] = (
            np.log1p(
                float(
                    data.get(
                        "weight",
                        1.0
                    )
                )
            )
        )

    # 初始布局
    try:

        pos0 = nx.kamada_kawai_layout(
            U,
            weight="layout_weight",
            scale=1.0
        )

    except Exception:

        pos0 = nx.circular_layout(
            U
        )

    # Spring 优化
    pos1 = nx.spring_layout(
        U,
        pos=pos0,
        weight="layout_weight",
        seed=LAYOUT_SEED + year,
        k=SPRING_K,
        iterations=SPRING_ITER,
        scale=1.0
    )

    # 轻度节点分离
    pos2 = separate_close_nodes(
        pos1,
        threshold=COLLISION_THRESHOLD,
        strength=COLLISION_STRENGTH,
        steps=COLLISION_STEPS
    )

    # 收紧外围节点
    pos3 = radial_compress_positions(
        pos2,
        compression=RADIAL_COMPRESSION
    )

    return pos3


def draw_top_network(
    G,
    year: int,
    output_dir: Path
):
    """
    绘制 weighted degree 最大的 50 个节点。
    """

    strength = weighted_total_degree(
        G
    )

    top_nodes = sorted(
        strength,
        key=strength.get,
        reverse=True
    )[
        :min(
            TOP_N,
            G.number_of_nodes()
        )
    ]

    H = G.subgraph(
        top_nodes
    ).copy()

    # 确保子图不存在自环
    H.remove_edges_from(
        list(
            nx.selfloop_edges(H)
        )
    )

    node_to_comm = get_communities(
        H
    )

    pos = compute_better_layout(
        H,
        year
    )

    # ---------------- 节点尺寸 ----------------

    node_strengths = np.array(
        [
            strength[n]
            for n in H.nodes()
        ],
        dtype=float
    )

    size_norm = robust_minmax(
        node_strengths,
        low=0.0,
        high=1.0
    )

    node_sizes = (
        280
        +
        1000
        *
        size_norm
    )

    # ---------------- 节点颜色 ----------------

    node_colors = [
        MACARON_COLORS[
            node_to_comm.get(
                n,
                0
            )
            %
            len(
                MACARON_COLORS
            )
        ]
        for n in H.nodes()
    ]

    # ---------------- 边宽 ----------------

    edge_weights = np.array(
        [
            float(
                data.get(
                    "weight",
                    1.0
                )
            )
            for _, _, data
            in H.edges(
                data=True
            )
        ],
        dtype=float
    )

    if len(
        edge_weights
    ):

        edge_norm = robust_minmax(
            edge_weights,
            low=0.0,
            high=1.0
        )

        edge_widths = (
            0.22
            +
            1.55
            *
            edge_norm
        )

        edge_alphas = (
            0.05
            +
            0.28
            *
            edge_norm
        )

    else:

        edge_widths = []
        edge_alphas = []

    # ---------------- 绘图 ----------------

    fig, ax = plt.subplots(
        figsize=(
            12.5,
            9.0
        )
    )

    # 逐边绘制
    for i, (
        u,
        v,
        data
    ) in enumerate(
        H.edges(
            data=True
        )
    ):

        nx.draw_networkx_edges(
            H,
            pos,
            edgelist=[
                (
                    u,
                    v
                )
            ],
            ax=ax,
            width=float(
                edge_widths[i]
            ),
            alpha=float(
                edge_alphas[i]
            ),
            edge_color="#6E7F93",
            arrows=G.is_directed(),
            arrowstyle="-|>",
            arrowsize=7,
            connectionstyle=(
                "arc3,rad=0.018"
                if G.is_directed()
                else
                "arc3"
            ),
            min_source_margin=9,
            min_target_margin=9,
        )

    # 节点
    nx.draw_networkx_nodes(
        H,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        edgecolors="#5C677D",
        linewidths=0.85,
        alpha=0.97,
    )

    # 标签
    label_nodes = sorted(
        H.nodes(),
        key=lambda n:
        strength[n],
        reverse=True
    )[
        :min(
            LABEL_TOP_N,
            H.number_of_nodes()
        )
    ]

    labels = {
        n: n
        for n in label_nodes
    }

    nx.draw_networkx_labels(
        H,
        pos,
        labels=labels,
        ax=ax,
        font_size=15,
        font_family="Times New Roman",
        font_color="#293241",
    )

    # 不显示 title
    ax.axis(
        "off"
    )

    ax.margins(
        0.08
    )

    # 手动布局
    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.025,
        top=0.975
    )

    out_path = (
        output_dir
        /
        f"network_top50_{year}_final.png"
    )

    fig.savefig(
        out_path,
        dpi=DPI,
        facecolor="white"
    )

    plt.close(
        fig
    )

    return (
        H,
        strength
    )


# ============================================================
# 4. 度分布
# ============================================================

def get_unweighted_degrees(
    G
):
    """
    获取全部节点的经典非加权度。
    """

    if not G.is_directed():

        return np.array(
            [
                d
                for _, d
                in G.degree()
            ],
            dtype=int
        )

    if DEGREE_MODE == "in":

        return np.array(
            [
                d
                for _, d
                in G.in_degree()
            ],
            dtype=int
        )

    if DEGREE_MODE == "out":

        return np.array(
            [
                d
                for _, d
                in G.out_degree()
            ],
            dtype=int
        )

    if DEGREE_MODE == "total":

        in_d = dict(
            G.in_degree()
        )

        out_d = dict(
            G.out_degree()
        )

        return np.array(
            [
                in_d[n]
                +
                out_d[n]
                for n in G.nodes()
            ],
            dtype=int
        )

    raise ValueError(
        f"DEGREE_MODE={DEGREE_MODE!r} 无效，"
        f"应为 'total' / 'in' / 'out'"
    )


def degree_distribution(
    G
):
    """
    计算全部节点的度分布：

        P(k) = N_k / N
    """

    degrees = get_unweighted_degrees(
        G
    )

    if len(
        degrees
    ) == 0:

        return (
            np.array([]),
            np.array([]),
            np.array([]),
            degrees
        )

    values, counts = np.unique(
        degrees,
        return_counts=True
    )

    # log 坐标中不能使用 k = 0
    mask = (
        values > 0
    )

    k = values[
        mask
    ].astype(
        float
    )

    counts_nonzero = counts[
        mask
    ].astype(
        float
    )

    pk = (
        counts_nonzero
        /
        len(
            degrees
        )
    )

    return (
        k,
        pk,
        counts_nonzero,
        degrees
    )


def fit_power_law_loglog(
    k,
    pk
):
    """
    对双对数坐标中的中间数据点做线性拟合。

    首先筛选：
        k > 0
        P(k) > 0

    然后按 k 从小到大排列。

    拟合时排除：
        第 1 个有效数据点；
        最后 5 个有效数据点。

    即：
        k_fit  = k[1:-5]
        pk_fit = pk[1:-5]

    最后拟合：
        log P(k) = a log k + b

    注意：
    Python 内部仍使用 log10 计算坐标，
    因为更换对数底不会改变直线斜率。
    图中统一显示为 log。
    """

    mask = (
        np.isfinite(k)
        &
        np.isfinite(pk)
        &
        (k > 0)
        &
        (pk > 0)
    )

    k_valid = np.asarray(
        k[mask],
        dtype=float
    )

    pk_valid = np.asarray(
        pk[mask],
        dtype=float
    )

    # 按 k 升序排列
    order = np.argsort(
        k_valid
    )

    k_valid = k_valid[
        order
    ]

    pk_valid = pk_valid[
        order
    ]

    n_valid = len(
        k_valid
    )

    min_required = (
        FIT_EXCLUDE_FIRST
        +
        FIT_EXCLUDE_LAST
        +
        3
    )

    if n_valid < min_required:

        print(
            f"警告：有效双对数点只有 {n_valid} 个，"
            f"排除前 {FIT_EXCLUDE_FIRST} 个和后 "
            f"{FIT_EXCLUDE_LAST} 个后不足 3 个点，"
            f"无法拟合。"
        )

        return None

    # --------------------------------------------------------
    # 核心修改：
    # 排除第一个点以及最后 5 个点
    # --------------------------------------------------------

    start_idx = FIT_EXCLUDE_FIRST

    end_idx = (
        n_valid
        -
        FIT_EXCLUDE_LAST
    )

    k_fit = k_valid[
        start_idx:end_idx
    ]

    pk_fit = pk_valid[
        start_idx:end_idx
    ]

    # 对数变换
    x = np.log10(
        k_fit
    )

    y = np.log10(
        pk_fit
    )

    if len(
        x
    ) < 3:

        return None

    res = linregress(
        x,
        y
    )

    return {
        "slope":
            res.slope,

        "intercept":
            res.intercept,

        "rvalue":
            res.rvalue,

        "r2":
            res.rvalue ** 2,

        "pvalue":
            res.pvalue,

        "stderr":
            res.stderr,

        # 用于统计输出
        "x":
            x,

        "y":
            y,

        # 保存真正参与拟合的原始 k 与 P(k)
        "k_fit":
            k_fit,

        "pk_fit":
            pk_fit,

        # 保存排除信息
        "excluded_first":
            FIT_EXCLUDE_FIRST,

        "excluded_last":
            FIT_EXCLUDE_LAST,

        "total_valid_points":
            n_valid,
    }


def draw_degree_distribution(
    G,
    year: int,
    output_dir: Path
):
    """
    单张图：

    外层：
        普通坐标系柱状度分布

    内嵌：
        双对数全部散点
        +
        基于中间点拟合得到的直线
    """

    (
        k,
        pk,
        counts,
        degrees
    ) = degree_distribution(
        G
    )

    fit = fit_power_law_loglog(
        k,
        pk
    )

    fig, ax = plt.subplots(
        figsize=(
            9.2,
            6.5
        )
    )

    # ========================================================
    # 外层主图：柱状图
    # ========================================================

    main_mask = (
        k <= MAIN_XMAX
    )

    k_main = k[
        main_mask
    ]

    pk_main = pk[
        main_mask
    ]

    ax.bar(
        k_main,
        pk_main,
        width=0.78,
        color="#A8DADC",
        edgecolor="#577590",
        linewidth=0.70,
        alpha=0.88,
        zorder=2
    )

    ax.set_xlim(
        0,
        MAIN_XMAX
    )

    ax.set_xlabel(
        r"Degree, $k$"
    )

    ax.set_ylabel(
        r"Probability, $P(k)$"
    )

    # 纵坐标约 4 个主刻度
    ax.yaxis.set_major_locator(
        MaxNLocator(
            nbins=4
        )
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=18
    )

    ax.grid(
        axis="y",
        alpha=0.16,
        linewidth=0.60,
        zorder=0
    )

    # ========================================================
    # 内嵌双对数图
    # ========================================================

    axins = inset_axes(
        ax,
        width="57%",
        height="61%",
        loc="upper right",
        borderpad=1.15
    )

    # 所有有效数据点都显示
    axins.scatter(
        k,
        pk,
        s=34,
        facecolor="#F7A8B8",
        edgecolor="#8D6A7B",
        linewidth=0.70,
        alpha=0.92,
        zorder=5
    )

    if fit is not None:

        # ----------------------------------------------------
        # 拟合线只覆盖真正参与拟合的数据范围
        # ----------------------------------------------------

        k_fit_min = fit["k_fit"].min()
        k_fit_max = fit["k_fit"].max()

        xfit = np.linspace(
            np.log10(
                k_fit_min
            ),
            np.log10(
                k_fit_max
            ),
            240
        )

        yfit = (
            fit["slope"]
            *
            xfit
            +
            fit["intercept"]
        )

        k_line = (
            10 ** xfit
        )

        pk_line = (
            10 ** yfit
        )

        axins.plot(
            k_line,
            pk_line,
            linestyle="--",
            linewidth=1.6,
            color="#4A5568",
            zorder=4
        )

        # p-value
        p_text = (
            r"$p<0.001$"
            if fit["pvalue"] < 0.001
            else
            rf"$p={fit['pvalue']:.4f}$"
        )

        # 图中统一使用 log
        equation = (
            rf"$\log P(k)"
            rf"={fit['slope']:.3f}\log k"
            rf"{fit['intercept']:+.3f}$"
        )

        stats = (
            equation
            +
            "\n"
            +
            rf"$R^2={fit['r2']:.4f}$"
            +
            "\n"
            +
            p_text
        )

        # ----------------------------------------------------
        # 增加三行文字之间的行间距
        # ----------------------------------------------------
        axins.text(
            0.95,
            0.94,
            stats,
            transform=axins.transAxes,
            ha="right",
            va="top",
            fontsize=16,
            linespacing=1.45,
            zorder=0,
            bbox=dict(
                boxstyle="round,pad=0.28",
                facecolor="white",
                edgecolor="#B0BEC5",
                alpha=0.72
            )
        )

    axins.set_xscale(
        "log"
    )

    axins.set_yscale(
        "log"
    )

    axins.set_xlabel(
        r"$k$",
        labelpad=1
    )

    axins.set_ylabel(
        r"$P(k)$",
        labelpad=1
    )

    axins.tick_params(
        axis="both",
        which="major",
        labelsize=18
    )

    axins.grid(
        which="both",
        alpha=0.13,
        linewidth=0.45
    )

    # 不显示 title

    # ========================================================
    # 手动留白
    # ========================================================

    fig.subplots_adjust(
        left=0.13,
        right=0.97,
        bottom=0.15,
        top=0.97
    )

    out_path = (
        output_dir
        /
        f"degree_distribution_{year}_final.png"
    )

    fig.savefig(
        out_path,
        dpi=DPI,
        facecolor="white"
    )

    plt.close(
        fig
    )

    # ========================================================
    # 保存度分布原始数据
    # ========================================================

    dist_df = pd.DataFrame({
        "degree_k":
            k.astype(int),

        "node_count":
            counts.astype(int),

        "P_k":
            pk,

        # 标记是否参与双对数拟合
        "used_in_fit":
            [
                (
                    fit is not None
                    and
                    np.any(
                        np.isclose(
                            kv,
                            fit["k_fit"]
                        )
                    )
                )
                for kv in k
            ]
    })

    dist_df.to_csv(
        output_dir
        /
        f"degree_distribution_{year}_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return (
        fit,
        degrees
    )


# ============================================================
# 5. 主程序
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fit_rows = []
    network_rows = []

    print(
        "=" * 72
    )

    print(
        "开始绘制供应链网络图和度分布图（最终修改版）"
    )

    print(
        f"数据路径：{DATA_DIR}"
    )

    print(
        f"输出路径：{OUTPUT_DIR}"
    )

    print(
        "=" * 72
    )

    for year in YEARS:

        print(
            f"\n>>> 正在处理 {year} 年 ..."
        )

        csv_path = (
            DATA_DIR
            /
            f"edgelist_{year}.csv"
        )

        # ----------------------------------------------------
        # 读取数据
        # ----------------------------------------------------

        df = read_network_csv(
            csv_path
        )

        print(
            f"有效边记录（已去自环）：{len(df):,}"
        )

        # ----------------------------------------------------
        # 构建网络
        # ----------------------------------------------------

        G = build_graph(
            df
        )

        print(
            f"网络规模："
            f"N={G.number_of_nodes():,}, "
            f"E={G.number_of_edges():,}"
        )

        # ----------------------------------------------------
        # 网络图
        # ----------------------------------------------------

        H, strength = draw_top_network(
            G,
            year,
            OUTPUT_DIR
        )

        print(
            "已输出网络图"
        )

        # ----------------------------------------------------
        # 度分布图
        # ----------------------------------------------------

        fit, degrees = draw_degree_distribution(
            G,
            year,
            OUTPUT_DIR
        )

        print(
            "已输出度分布图"
        )

        # ----------------------------------------------------
        # 网络统计
        # ----------------------------------------------------

        network_rows.append({
            "year":
                year,

            "nodes":
                G.number_of_nodes(),

            "edges":
                G.number_of_edges(),

            "mean_degree":
                (
                    float(
                        np.mean(
                            degrees
                        )
                    )
                    if len(
                        degrees
                    )
                    else np.nan
                ),

            "median_degree":
                (
                    float(
                        np.median(
                            degrees
                        )
                    )
                    if len(
                        degrees
                    )
                    else np.nan
                ),

            "max_degree":
                (
                    int(
                        np.max(
                            degrees
                        )
                    )
                    if len(
                        degrees
                    )
                    else np.nan
                ),

            "density":
                nx.density(
                    G
                ),

            "degree_mode":
                DEGREE_MODE,

            "directed":
                G.is_directed(),
        })

        # ----------------------------------------------------
        # 双对数拟合统计
        # ----------------------------------------------------

        if fit is not None:

            fit_rows.append({
                "year":
                    year,

                "slope":
                    fit["slope"],

                "power_law_exponent_alpha_approx":
                    -fit["slope"],

                "intercept_log":
                    fit["intercept"],

                "R_squared":
                    fit["r2"],

                "p_value":
                    fit["pvalue"],

                "std_err":
                    fit["stderr"],

                "total_valid_points":
                    fit["total_valid_points"],

                "excluded_first_points":
                    fit["excluded_first"],

                "excluded_last_points":
                    fit["excluded_last"],

                "fit_points":
                    len(
                        fit["x"]
                    ),

                "fit_k_min":
                    float(
                        fit["k_fit"].min()
                    ),

                "fit_k_max":
                    float(
                        fit["k_fit"].max()
                    ),
            })

            print(
                f"Log-log 拟合："
                f"排除第 {FIT_EXCLUDE_FIRST} 个前端点，"
                f"排除最后 {FIT_EXCLUDE_LAST} 个点；"
                f"实际拟合点数={len(fit['x'])}"
            )

            print(
                f"拟合范围："
                f"k={fit['k_fit'].min():.0f}"
                f" ~ "
                f"{fit['k_fit'].max():.0f}"
            )

            print(
                f"slope={fit['slope']:.4f}, "
                f"R^2={fit['r2']:.4f}, "
                f"p={fit['pvalue']:.3e}"
            )

        else:

            fit_rows.append({
                "year":
                    year,

                "slope":
                    np.nan,

                "power_law_exponent_alpha_approx":
                    np.nan,

                "intercept_log":
                    np.nan,

                "R_squared":
                    np.nan,

                "p_value":
                    np.nan,

                "std_err":
                    np.nan,

                "total_valid_points":
                    np.nan,

                "excluded_first_points":
                    FIT_EXCLUDE_FIRST,

                "excluded_last_points":
                    FIT_EXCLUDE_LAST,

                "fit_points":
                    0,

                "fit_k_min":
                    np.nan,

                "fit_k_max":
                    np.nan,
            })

            print(
                "警告：有效度值太少，无法完成 log-log 拟合。"
            )

    # ========================================================
    # 保存网络统计汇总
    # ========================================================

    pd.DataFrame(
        network_rows
    ).to_csv(
        OUTPUT_DIR
        /
        "network_summary_4years_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 保存双对数拟合汇总
    # ========================================================

    pd.DataFrame(
        fit_rows
    ).to_csv(
        OUTPUT_DIR
        /
        "powerlaw_loglog_fit_4years_final.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "全部完成。"
    )

    print(
        "输出文件："
    )

    for year in YEARS:

        print(
            f"  network_top50_{year}_final.png"
        )

        print(
            f"  degree_distribution_{year}_final.png"
        )

        print(
            f"  degree_distribution_{year}_final.csv"
        )

    print(
        "  network_summary_4years_final.csv"
    )

    print(
        "  powerlaw_loglog_fit_4years_final.csv"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()
