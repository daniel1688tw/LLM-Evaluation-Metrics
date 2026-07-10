"""
結果視覺化模組。

本模組負責讀取三個評估任務（QA、文本摘要、RAG）已產生的真實結果 CSV 檔案，
並將各模型的評估指標畫成長條圖，方便比較不同模型的表現。

重要原則：
- 本模組只讀取既有的 results/*.csv 檔案，絕對不生成、模擬或假造任何評估數字。
- 若指定的 CSV 檔案不存在，會印出清楚的警告訊息並跳過該圖表，不會用假資料頂替。
"""

import os

import matplotlib
matplotlib.use("Agg")  # 無圖形介面環境下也能存檔
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd

DPI = 300

# 常見的中文字型檔案路徑（容器內若有掛載 Windows 字型目錄，或系統本身安裝了中文字型，
# 皆可在此清單中被找到並註冊給 matplotlib 使用，讓中文標題與軸標籤能正確顯示）。
# 若清單中的路徑都找不到，則印出警告並沿用預設字型（僅影響圖片中中文字的顯示效果，
# 不影響任何數據內容的正確性）。
_CJK_FONT_CANDIDATES = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # apt 套件 fonts-noto-cjk 安裝路徑（見 Dockerfile）
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf",
    "/usr/share/fonts/host_win/msjh.ttc",  # 掛載 Windows 字型目錄時的路徑（Microsoft JhengHei）
    "C:/Windows/Fonts/msjh.ttc",
    "C:/Windows/Fonts/msjhbd.ttc",
]


def _setup_cjk_font() -> None:
    """
    嘗試從候選路徑中找到一個可用的中文字型檔案並註冊給 matplotlib 使用，
    讓圖表中的繁體中文標題與軸標籤能正確顯示（而非顯示為缺字方框）。

    若找不到任何候選字型，僅印出警告，不影響圖表的產生或資料正確性。
    """
    for font_path in _CJK_FONT_CANDIDATES:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            font_name = fm.FontProperties(fname=font_path).get_name()
            matplotlib.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return

    print("警告：找不到可用的中文字型檔案，圖表中的中文文字可能無法正常顯示（不影響數據正確性）。")
    matplotlib.rcParams["axes.unicode_minus"] = False


_setup_cjk_font()


def _bar_subplot(ax, df: pd.DataFrame, column: str, title: str) -> None:
    """
    在指定的座標軸上畫出單一指標欄位的長條圖。

    參數：
        ax: matplotlib 的座標軸物件
        df: 包含評估結果的 DataFrame
        column: 要繪製的欄位名稱
        title: 子圖標題（繁體中文）
    """
    ax.bar(df["Model"], df[column], color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("模型")
    ax.set_ylabel(column)
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")


def plot_qa_results(results_csv: str = "results/qa_results.csv"):
    """
    讀取 QA 任務的真實評估結果 CSV，畫出 Exact Match 與 F1 Score 的對比長條圖。

    參數：
        results_csv: QA 結果 CSV 檔案路徑

    回傳：
        matplotlib 的 Figure 物件
    """
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    _bar_subplot(axes[0], df, "Exact Match (%)", "QA 任務：Exact Match")
    _bar_subplot(axes[1], df, "F1 Score (%)", "QA 任務：F1 Score")

    fig.suptitle("QA 任務模型對比（真實評估結果）")
    fig.tight_layout()
    return fig


def plot_summarization_results(results_csv: str = "results/summarization_results.csv"):
    """
    讀取文本摘要任務的真實評估結果 CSV，
    畫出 ROUGE-1、ROUGE-2、ROUGE-L、BERTScore 四個指標的對比長條圖（2x2 子圖）。

    參數：
        results_csv: 摘要結果 CSV 檔案路徑

    回傳：
        matplotlib 的 Figure 物件
    """
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    _bar_subplot(axes[0, 0], df, "ROUGE-1 (%)", "文本摘要任務：ROUGE-1")
    _bar_subplot(axes[0, 1], df, "ROUGE-2 (%)", "文本摘要任務：ROUGE-2")
    _bar_subplot(axes[1, 0], df, "ROUGE-L (%)", "文本摘要任務：ROUGE-L")
    _bar_subplot(axes[1, 1], df, "BERTScore (%)", "文本摘要任務：BERTScore")

    fig.suptitle("文本摘要任務模型對比（真實評估結果）")
    fig.tight_layout()
    return fig


def plot_rag_results(results_csv: str = "results/rag_results.csv"):
    """
    讀取 RAG 任務的真實評估結果 CSV，
    畫出 Hit@1、Hit@5、Hit@10、MRR、File-level Recall@10 五個指標的對比長條圖（2x3 子圖，留一格空白）。

    參數：
        results_csv: RAG 結果 CSV 檔案路徑

    回傳：
        matplotlib 的 Figure 物件
    """
    df = pd.read_csv(results_csv)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    _bar_subplot(axes[0, 0], df, "Hit@1 (%)", "RAG 任務：Hit@1")
    _bar_subplot(axes[0, 1], df, "Hit@5 (%)", "RAG 任務：Hit@5")
    _bar_subplot(axes[0, 2], df, "Hit@10 (%)", "RAG 任務：Hit@10")
    _bar_subplot(axes[1, 0], df, "MRR (%)", "RAG 任務：MRR")
    _bar_subplot(axes[1, 1], df, "File-level Recall@10 (%)", "RAG 任務：File-level Recall@10")
    axes[1, 2].axis("off")  # 留一格空白

    fig.suptitle("RAG 任務模型對比（真實評估結果）")
    fig.tight_layout()
    return fig


def save_all_plots(results_dir: str = "results") -> None:
    """
    針對三個評估任務，若對應的 CSV 存在則畫圖並存檔；若不存在則印出警告並跳過。

    絕對不會用假資料替代缺失的 CSV。

    參數：
        results_dir: 存放結果 CSV 與輸出圖檔的目錄
    """
    tasks = [
        ("qa_results.csv", "qa_comparison.png", plot_qa_results, "QA"),
        (
            "summarization_results.csv",
            "summarization_comparison.png",
            plot_summarization_results,
            "文本摘要",
        ),
        ("rag_results.csv", "rag_comparison.png", plot_rag_results, "RAG"),
    ]

    for csv_name, png_name, plot_fn, task_label in tasks:
        csv_path = os.path.join(results_dir, csv_name)
        if not os.path.exists(csv_path):
            print(f"警告：找不到 {csv_path}，跳過 {task_label} 任務的圖表繪製（不使用假資料）。")
            continue

        fig = plot_fn(csv_path)
        png_path = os.path.join(results_dir, png_name)
        fig.savefig(png_path, dpi=DPI)
        plt.close(fig)
        print(f"已產生圖表：{png_path}")


def main() -> None:
    """主程式進入點：畫出所有可用的評估結果對比圖。"""
    save_all_plots()
    print("視覺化作業完成（僅使用 results/ 目錄下實際存在的真實評估結果 CSV）。")


if __name__ == "__main__":
    main()
