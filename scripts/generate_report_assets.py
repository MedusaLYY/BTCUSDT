from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


def _bootstrap_venv_if_available() -> None:
    """Re-run with the project virtualenv before importing plotting dependencies."""
    if os.environ.get("REPORT_ASSET_BOOTSTRAPPED") == "1":
        return
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return
    current_python = Path(sys.executable).resolve()
    target_python = venv_python.resolve()
    if current_python == target_python:
        return
    os.environ["REPORT_ASSET_BOOTSTRAPPED"] = "1"
    os.execv(str(target_python), [str(target_python), str(SCRIPT_PATH), *sys.argv[1:]])


_bootstrap_venv_if_available()

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from features.build_features import build_feature_frame  # noqa: E402
from labels.build_labels import add_future_labels  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "config" / "training.yaml"
TRAINING_METRICS_PATH = PROJECT_ROOT / "runs" / "latest" / "reports" / "training_metrics.json"
TEST_PREDICTIONS_PATH = PROJECT_ROOT / "runs" / "latest" / "outputs" / "test_predictions.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "runs" / "latest" / "models" / "feature_columns.json"
ASSETS_DIR = PROJECT_ROOT / "docs" / "assets"

REQUIRED_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
]

REQUIRED_TEST_PREDICTION_COLUMNS = [
    "open_time",
    "buy_probability",
    "predicted_max_return",
    "future_max_return_30m",
    "signal",
]


def main() -> None:
    _configure_matplotlib()
    config = _load_yaml(CONFIG_PATH)
    metrics = _load_json(TRAINING_METRICS_PATH)
    feature_columns = _load_json(FEATURE_COLUMNS_PATH)
    dataset_path = _resolve_project_path(config["dataset_path"])

    raw = _read_csv(dataset_path, REQUIRED_KLINE_COLUMNS)
    _read_csv(TEST_PREDICTIONS_PATH, REQUIRED_TEST_PREDICTION_COLUMNS)

    if not isinstance(feature_columns, list) or not all(
        isinstance(column, str) for column in feature_columns
    ):
        raise ValueError(f"{FEATURE_COLUMNS_PATH} 必须是字符串列表。")

    raw = _normalize_kline_frame(raw)
    label_config = config.get("label", {})
    labeled = add_future_labels(
        raw,
        horizon=int(label_config.get("horizon", 6)),
        upside_threshold=float(label_config.get("upside_threshold", 0.002)),
    )
    featured, generated_feature_columns = build_feature_frame(labeled)

    missing_saved_features = sorted(set(feature_columns) - set(featured.columns))
    if missing_saved_features:
        raise ValueError(
            f"特征工程结果缺少 feature_columns.json 中的列: {missing_saved_features}"
        )

    heatmap_features = [column for column in feature_columns if column in generated_feature_columns]
    if not heatmap_features:
        raise ValueError("feature_columns.json 与特征工程输出没有可用于相关性分析的交集。")

    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    generated_paths = [
        _plot_close_price(raw, ASSETS_DIR / "close_price_timeseries.png"),
        _plot_return_distribution(raw, ASSETS_DIR / "return_1_distribution.png"),
        _plot_volume_distribution(raw, ASSETS_DIR / "volume_distribution.png"),
        _plot_future_max_return_distribution(
            labeled, ASSETS_DIR / "future_max_return_distribution.png"
        ),
        _plot_y_buy_distribution(labeled, ASSETS_DIR / "y_buy_distribution.png"),
        _plot_monthly_positive_rate(labeled, ASSETS_DIR / "monthly_positive_rate.png"),
        _plot_feature_correlation_heatmap(
            featured,
            heatmap_features,
            ASSETS_DIR / "feature_correlation_heatmap.png",
        ),
        _plot_time_split_chart(metrics, ASSETS_DIR / "time_split_chart.png"),
        _plot_lightgbm_dual_model_structure(
            ASSETS_DIR / "lightgbm_dual_model_structure.png"
        ),
        _plot_model_training_flow(ASSETS_DIR / "model_training_flow.png"),
        _plot_system_architecture(ASSETS_DIR / "system_architecture.png"),
        _plot_data_storage_structure(ASSETS_DIR / "data_storage_structure.png"),
        _plot_realtime_prediction_flow(ASSETS_DIR / "realtime_prediction_flow.png"),
        _plot_delayed_settlement_flow(ASSETS_DIR / "delayed_settlement_flow.png"),
        _plot_github_vercel_deployment_architecture(
            ASSETS_DIR / "github_vercel_deployment_architecture.png"
        ),
    ]

    print("报告图表已生成到 docs/assets：")
    for path in generated_paths:
        print(f"- {path.relative_to(PROJECT_ROOT)}")


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["savefig.dpi"] = 180


def _load_yaml(path: Path) -> dict:
    _require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} 内容必须是 YAML 对象。")
    return data


def _load_json(path: Path):
    _require_file(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path, required_columns: Iterable[str]) -> pd.DataFrame:
    _require_file(path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"读取 CSV 失败: {path}，原因: {exc}") from exc
    _require_columns(frame, required_columns, path)
    return frame


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"缺少必要文件: {path}")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} 缺少必要列: {missing}")


def _resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _normalize_kline_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["open_time"] = pd.to_datetime(normalized["open_time"], errors="coerce")
    if normalized["open_time"].isna().any():
        bad_count = int(normalized["open_time"].isna().sum())
        raise ValueError(f"open_time 存在无法解析的时间值，共 {bad_count} 行。")

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    ]
    for column in numeric_columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    numeric_missing = normalized[numeric_columns].isna().sum()
    numeric_missing = numeric_missing[numeric_missing > 0]
    if not numeric_missing.empty:
        raise ValueError(f"K线数值列存在缺失或无法解析值: {numeric_missing.to_dict()}")

    return normalized.sort_values("open_time").reset_index(drop=True)


def _save_current_figure(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    return path


def _plot_close_price(frame: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["open_time"], frame["close"], linewidth=0.8, color="#1F4E79")
    ax.set_title("BTCUSDT 收盘价时间序列")
    ax.set_xlabel("时间")
    ax.set_ylabel("收盘价（USDT）")
    ax.grid(alpha=0.25)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return _save_current_figure(path)


def _plot_return_distribution(frame: pd.DataFrame, path: Path) -> Path:
    returns = frame["close"].pct_change(fill_method=None).dropna()
    lower, upper = returns.quantile([0.001, 0.999])
    clipped = returns.clip(lower=lower, upper=upper)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clipped, bins=120, color="#548235", alpha=0.82)
    ax.set_title("5分钟收益率分布（按0.1%和99.9%分位截尾显示）")
    ax.set_xlabel("5分钟收益率")
    ax.set_ylabel("频数")
    ax.grid(alpha=0.25)
    return _save_current_figure(path)


def _plot_volume_distribution(frame: pd.DataFrame, path: Path) -> Path:
    volume = frame["volume"].astype(float)
    log_volume = np.log10(volume[volume > 0])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(log_volume, bins=100, color="#8064A2", alpha=0.82)
    ax.set_title("成交量 log10 分布")
    ax.set_xlabel("log10(volume)")
    ax.set_ylabel("频数")
    ax.grid(alpha=0.25)
    return _save_current_figure(path)


def _plot_future_max_return_distribution(frame: pd.DataFrame, path: Path) -> Path:
    _require_columns(frame, ["future_max_return_30m"], Path("labeled_frame"))
    values = frame["future_max_return_30m"].dropna()
    lower, upper = values.quantile([0.001, 0.999])
    clipped = values.clip(lower=lower, upper=upper)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(clipped, bins=120, color="#F79646", alpha=0.82)
    ax.axvline(0.002, color="#C00000", linestyle="--", linewidth=1.4, label="y_buy阈值：0.2%")
    ax.set_title("future_max_return_30m 分布")
    ax.set_xlabel("未来30分钟最大涨幅")
    ax.set_ylabel("频数")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save_current_figure(path)


def _plot_y_buy_distribution(frame: pd.DataFrame, path: Path) -> Path:
    _require_columns(frame, ["y_buy"], Path("labeled_frame"))
    counts = frame["y_buy"].astype(int).value_counts().reindex([0, 1], fill_value=0)
    labels = ["负样本 y_buy=0", "正样本 y_buy=1"]
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=["#7F7F7F", "#1F4E79"], alpha=0.86)
    for bar, count in zip(bars, counts.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(count)}\n{count / total:.2%}",
            ha="center",
            va="bottom",
        )
    ax.set_title("y_buy 正负样本比例")
    ax.set_ylabel("样本数")
    ax.grid(axis="y", alpha=0.25)
    return _save_current_figure(path)


def _plot_monthly_positive_rate(frame: pd.DataFrame, path: Path) -> Path:
    _require_columns(frame, ["open_time", "y_buy"], Path("labeled_frame"))
    monthly = frame.groupby(frame["open_time"].dt.to_period("M"))["y_buy"].mean()
    x_values = monthly.index.astype(str)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x_values, monthly.values, marker="o", linewidth=1.6, color="#1F4E79")
    ax.set_title("月度正样本率变化")
    ax.set_xlabel("月份")
    ax.set_ylabel("正样本率")
    ax.set_ylim(0, max(0.65, float(monthly.max()) + 0.05))
    ax.grid(alpha=0.25)
    ax.tick_params(axis="x", rotation=45)
    return _save_current_figure(path)


def _plot_feature_correlation_heatmap(
    frame: pd.DataFrame,
    feature_columns: list[str],
    path: Path,
) -> Path:
    heatmap_columns = feature_columns + ["future_max_return_30m", "y_buy"]
    _require_columns(frame, heatmap_columns, Path("featured_frame"))
    corr = frame[heatmap_columns].corr(numeric_only=True)

    fig_width = max(12, len(heatmap_columns) * 0.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.78))
    image = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("特征相关性热力图")
    ax.set_xticks(range(len(heatmap_columns)))
    ax.set_yticks(range(len(heatmap_columns)))
    ax.set_xticklabels(heatmap_columns, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(heatmap_columns, fontsize=8)
    color_bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    color_bar.set_label("Pearson相关系数")
    return _save_current_figure(path)


def _plot_time_split_chart(metrics: dict, path: Path) -> Path:
    split = metrics.get("split")
    if not isinstance(split, dict):
        raise ValueError(f"{TRAINING_METRICS_PATH} 缺少 split 字段。")

    rows = []
    for name, label, color in [
        ("train", "训练集", "#1F4E79"),
        ("valid", "验证集", "#F79646"),
        ("test", "测试集", "#548235"),
    ]:
        item = split.get(name)
        if not item:
            raise ValueError(f"{TRAINING_METRICS_PATH} 的 split 缺少 {name}。")
        rows.append(
            {
                "label": label,
                "rows": int(item["rows"]),
                "start": pd.Timestamp(item["time_start"]),
                "end": pd.Timestamp(item["time_end"]),
                "color": color,
            }
        )

    fig, ax = plt.subplots(figsize=(12, 3.8))
    for idx, row in enumerate(rows):
        start_num = mdates.date2num(row["start"])
        end_num = mdates.date2num(row["end"])
        ax.barh(
            idx,
            end_num - start_num,
            left=start_num,
            height=0.45,
            color=row["color"],
            alpha=0.86,
        )
        ax.text(
            start_num + (end_num - start_num) / 2,
            idx,
            f"{row['label']}：{row['rows']}条",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row["label"] for row in rows])
    ax.set_title("训练/验证/测试时间切分图")
    ax.set_xlabel("时间")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="x", alpha=0.25)
    fig.autofmt_xdate()
    return _save_current_figure(path)


def _plot_lightgbm_dual_model_structure(path: Path) -> Path:
    nodes = {
        "data": ("BTCUSDT 5分钟K线\nopen/high/low/close/volume", (0.12, 0.72)),
        "features": ("历史窗口特征\n收益率/均线/波动/成交量/K线形态", (0.34, 0.72)),
        "clf": ("LightGBMClassifier\n输出 buy_probability", (0.58, 0.84)),
        "reg": ("LightGBMRegressor\n输出 predicted_max_return", (0.58, 0.58)),
        "engine": ("信号规则引擎\nBUY / WATCH / NO_BUY", (0.82, 0.72)),
    }
    edges = [
        ("data", "features"),
        ("features", "clf"),
        ("features", "reg"),
        ("clf", "engine"),
        ("reg", "engine"),
    ]
    return _plot_flow_diagram(
        "LightGBM 双模型结构图",
        nodes,
        edges,
        path,
        box_width=0.18,
        box_height=0.16,
    )


def _plot_system_architecture(path: Path) -> Path:
    nodes = {
        "binance": ("Binance K线数据\n历史/实时5分钟K线", (0.12, 0.74)),
        "backend": ("后端 API 服务\n行情、信号、回测、实时指标", (0.36, 0.74)),
        "models": ("LightGBM 模型文件\nbuy_classifier / return_regressor", (0.36, 0.43)),
        "storage": ("CSV 产物存储\npredictions / reports / metrics", (0.62, 0.43)),
        "frontend": ("前端 Dashboard\n当前行情、信号、实时记录", (0.62, 0.74)),
        "user": ("用户浏览器\n课程展示与结果查看", (0.86, 0.74)),
    }
    edges = [
        ("binance", "backend"),
        ("models", "backend"),
        ("backend", "storage"),
        ("backend", "frontend"),
        ("frontend", "user"),
    ]
    return _plot_flow_diagram(
        "系统总体架构图",
        nodes,
        edges,
        path,
        box_width=0.19,
        box_height=0.15,
    )


def _plot_model_training_flow(path: Path) -> Path:
    nodes = {
        "raw": ("原始 BTCUSDT 5分钟K线\n数据审计与时间排序", (0.11, 0.70)),
        "label": ("标签构造\nfuture_max_return_30m / y_buy", (0.31, 0.70)),
        "feature": ("历史特征工程\n20个固定特征列", (0.51, 0.70)),
        "split": ("时间顺序切分\ntrain / valid / test，gap=6", (0.71, 0.70)),
        "train": ("LightGBM 双模型训练\n分类器 + 回归器", (0.89, 0.70)),
        "eval": ("测试集评估与研究型回测\nmetrics / predictions / reports", (0.71, 0.38)),
        "artifact": ("保存模型产物\nmodels / feature_columns / metadata", (0.89, 0.38)),
    }
    edges = [
        ("raw", "label"),
        ("label", "feature"),
        ("feature", "split"),
        ("split", "train"),
        ("train", "eval"),
        ("train", "artifact"),
        ("eval", "artifact"),
    ]
    return _plot_flow_diagram(
        "模型训练流程图",
        nodes,
        edges,
        path,
        box_width=0.16,
        box_height=0.14,
    )


def _plot_data_storage_structure(path: Path) -> Path:
    nodes = {
        "dataset": ("数据集/\nBTCUSDT_5m_2y.csv", (0.18, 0.78)),
        "config": ("config/\ntraining.yaml", (0.18, 0.52)),
        "models": ("runs/latest/models/\nbuy_classifier.txt\nreturn_regressor.txt\nfeature_columns.json", (0.47, 0.78)),
        "reports": ("runs/latest/reports/\ntraining_metrics.json\nbacktest_report.json\nlive_metrics.json", (0.47, 0.52)),
        "outputs": ("runs/latest/outputs/\ntest_predictions.csv\nlive_predictions.csv", (0.47, 0.26)),
        "api": ("后端 API / Dashboard\n读取模型、报告与预测记录", (0.78, 0.52)),
    }
    edges = [
        ("dataset", "reports"),
        ("config", "reports"),
        ("models", "api"),
        ("reports", "api"),
        ("outputs", "api"),
    ]
    return _plot_flow_diagram(
        "数据存储结构图",
        nodes,
        edges,
        path,
        box_width=0.22,
        box_height=0.17,
    )


def _plot_realtime_prediction_flow(path: Path) -> Path:
    nodes = {
        "fetch": ("获取最新已收盘K线", (0.11, 0.70)),
        "feature": ("构造当前历史特征", (0.31, 0.70)),
        "load": ("加载特征列与模型文件", (0.51, 0.70)),
        "predict": ("输出概率与最大涨幅预测", (0.71, 0.70)),
        "signal": ("生成信号并返回前端", (0.89, 0.70)),
        "record": ("写入 live_predictions\n状态为 PENDING", (0.71, 0.38)),
        "dashboard": ("Dashboard 展示\n最新信号与记录", (0.89, 0.38)),
    }
    edges = [
        ("fetch", "feature"),
        ("feature", "load"),
        ("load", "predict"),
        ("predict", "signal"),
        ("signal", "dashboard"),
        ("predict", "record"),
        ("record", "dashboard"),
    ]
    return _plot_flow_diagram(
        "实时预测流程图",
        nodes,
        edges,
        path,
        box_width=0.16,
        box_height=0.14,
    )


def _plot_delayed_settlement_flow(path: Path) -> Path:
    nodes = {
        "pending": ("PENDING 预测记录\n不参与实时准确率", (0.13, 0.70)),
        "wait": ("等待未来6根5分钟K线\n约30分钟窗口完整", (0.37, 0.70)),
        "actual": ("计算 actual_future_max_return_30m\n与实际 y_buy", (0.62, 0.70)),
        "settled": ("更新为 SETTLED\n写入命中与误差字段", (0.86, 0.70)),
        "metrics": ("仅基于 SETTLED 记录\n计算 live_metrics", (0.62, 0.39)),
        "front": ("前端实时准确率卡片\n展示结算后统计", (0.86, 0.39)),
    }
    edges = [
        ("pending", "wait"),
        ("wait", "actual"),
        ("actual", "settled"),
        ("settled", "metrics"),
        ("metrics", "front"),
    ]
    return _plot_flow_diagram(
        "PENDING/SETTLED 延迟结算机制图",
        nodes,
        edges,
        path,
        box_width=0.19,
        box_height=0.15,
    )


def _plot_github_vercel_deployment_architecture(path: Path) -> Path:
    nodes = {
        "github": ("GitHub 仓库\n代码托管与版本管理", (0.16, 0.78)),
        "vercel": ("Vercel 前端 Dashboard\nRoot Directory: frontend", (0.42, 0.78)),
        "browser": ("用户浏览器\n访问在线页面", (0.72, 0.78)),
        "api": ("独立后端 API 服务\n本地或云服务器", (0.42, 0.43)),
        "model": ("LightGBM 模型文件", (0.16, 0.22)),
        "binance": ("Binance 实时K线", (0.42, 0.22)),
        "live": ("live_predictions 存储\nCSV 或数据库", (0.68, 0.22)),
    }
    edges = [
        ("github", "vercel"),
        ("vercel", "browser"),
        ("browser", "api"),
        ("api", "model"),
        ("api", "binance"),
        ("api", "live"),
    ]
    return _plot_flow_diagram(
        "GitHub + Vercel 部署架构图",
        nodes,
        edges,
        path,
        box_width=0.2,
        box_height=0.15,
    )


def _plot_flow_diagram(
    title: str,
    nodes: dict[str, tuple[str, tuple[float, float]]],
    edges: list[tuple[str, str]],
    path: Path,
    *,
    box_width: float,
    box_height: float,
) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=18)

    for key, (label, (x_pos, y_pos)) in nodes.items():
        face_color = "#EAF2F8" if key not in {"models", "storage", "model", "binance", "live"} else "#F4F6F6"
        patch = FancyBboxPatch(
            (x_pos - box_width / 2, y_pos - box_height / 2),
            box_width,
            box_height,
            boxstyle="round,pad=0.018,rounding_size=0.018",
            linewidth=1.2,
            edgecolor="#1F4E79",
            facecolor=face_color,
        )
        ax.add_patch(patch)
        ax.text(
            x_pos,
            y_pos,
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#17365D",
            linespacing=1.35,
        )

    for start_key, end_key in edges:
        start_x, start_y = nodes[start_key][1]
        end_x, end_y = nodes[end_key][1]
        arrow = FancyArrowPatch(
            (start_x, start_y),
            (end_x, end_y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.3,
            color="#595959",
            shrinkA=45,
            shrinkB=45,
            connectionstyle="arc3,rad=0.0",
        )
        ax.add_patch(arrow)

    return _save_current_figure(path)


if __name__ == "__main__":
    main()
