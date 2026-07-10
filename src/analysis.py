"""
結果分析模組。

本模組負責讀取三個評估任務（QA、文本摘要、RAG）已產生的真實結果 CSV 檔案，
並產生一份繁體中文的彙總分析報告。

重要原則：
- 本模組只讀取既有的 results/*.csv 檔案，絕對不生成、模擬或假造任何評估數字。
- 若指定的 CSV 檔案不存在，報告中該段落會註明「結果尚未產生」，不會編造任何數字。
"""

import os

import pandas as pd

QA_CSV = "results/qa_results.csv"
SUMMARIZATION_CSV = "results/summarization_results.csv"
RAG_CSV = "results/rag_results.csv"
REPORT_PATH = "results/analysis_report.txt"


def _read_csv_if_exists(csv_path: str):
    """
    若 CSV 檔案存在則讀取並回傳 DataFrame，否則回傳 None。

    參數：
        csv_path: CSV 檔案路徑

    回傳：
        pandas.DataFrame 或 None（檔案不存在時）
    """
    if not os.path.exists(csv_path):
        return None
    return pd.read_csv(csv_path)


def _analyze_qa(csv_path: str = QA_CSV) -> str:
    """
    分析 QA 任務的真實評估結果，回傳繁體中文分析文字段落。

    參數：
        csv_path: QA 結果 CSV 檔案路徑

    回傳：
        分析報告字串
    """
    lines = ["## QA 任務分析", ""]
    df = _read_csv_if_exists(csv_path)

    if df is None:
        lines.append("結果尚未產生（找不到 results/qa_results.csv）。")
        lines.append("")
        return "\n".join(lines)

    lines.append("結果表格：")
    lines.append(df.to_string(index=False))
    lines.append("")

    best_em_row = df.loc[df["Exact Match (%)"].idxmax()]
    best_f1_row = df.loc[df["F1 Score (%)"].idxmax()]

    lines.append(
        f"Exact Match 最高的模型為「{best_em_row['Model']}」"
        f"（{best_em_row['Exact Match (%)']:.2f}%）。"
    )
    lines.append(
        f"F1 Score 最高的模型為「{best_f1_row['Model']}」"
        f"（{best_f1_row['F1 Score (%)']:.2f}%）。"
    )
    lines.append("")
    lines.append(
        "指標說明：Exact Match（完全匹配）是嚴格度量，"
        "只有當模型輸出與標準答案逐字完全相同時才計分，任何字詞差異都會判定為錯誤；"
        "F1 Score 則是詞元（token）級別的部分匹配分數，"
        "會計算預測答案與標準答案在詞元重疊上的精確率與召回率的調和平均，"
        "屬於較寬鬆的評分方式，能反映答案「部分正確」的情況。"
    )
    lines.append("")
    return "\n".join(lines)


def _analyze_summarization(csv_path: str = SUMMARIZATION_CSV) -> str:
    """
    分析文本摘要任務的真實評估結果，回傳繁體中文分析文字段落。

    參數：
        csv_path: 摘要結果 CSV 檔案路徑

    回傳：
        分析報告字串
    """
    lines = ["## 文本摘要任務分析", ""]
    df = _read_csv_if_exists(csv_path)

    if df is None:
        lines.append("結果尚未產生（找不到 results/summarization_results.csv）。")
        lines.append("")
        return "\n".join(lines)

    lines.append("結果表格：")
    lines.append(df.to_string(index=False))
    lines.append("")

    best_rouge1_row = df.loc[df["ROUGE-1 (%)"].idxmax()]
    lines.append(
        f"ROUGE-1 最高的模型為「{best_rouge1_row['Model']}」"
        f"（{best_rouge1_row['ROUGE-1 (%)']:.2f}%）。"
    )
    lines.append("")
    lines.append(
        "指標說明：ROUGE 系列（ROUGE-1、ROUGE-2、ROUGE-L）是以 n-gram 重疊為基礎的指標，"
        "分別衡量生成摘要與參考摘要在單一詞（unigram）、雙詞（bigram）"
        "以及最長共同子序列（Longest Common Subsequence）上的重疊程度，"
        "屬於表面詞彙層級的比對，無法判斷語意是否相近但用詞不同的情況；"
        "BERTScore 則是利用預訓練語言模型的詞向量計算生成摘要與參考摘要之間的語義相似度，"
        "即使用詞不完全相同，只要語意相近也能獲得較高分數，"
        "因此能補足 ROUGE 系列在語義層面的不足。"
    )
    lines.append("")
    return "\n".join(lines)


def _analyze_rag(csv_path: str = RAG_CSV) -> str:
    """
    分析 RAG 任務的真實評估結果，回傳繁體中文分析文字段落。

    參數：
        csv_path: RAG 結果 CSV 檔案路徑

    回傳：
        分析報告字串
    """
    lines = ["## RAG 任務分析", ""]
    df = _read_csv_if_exists(csv_path)

    if df is None:
        lines.append("結果尚未產生（找不到 results/rag_results.csv）。")
        lines.append("")
        return "\n".join(lines)

    lines.append("結果表格：")
    lines.append(df.to_string(index=False))
    lines.append("")

    best_hit10_row = df.loc[df["Hit@10 (%)"].idxmax()]
    lines.append(
        f"Hit@10 最高的模型為「{best_hit10_row['Model']}」"
        f"（{best_hit10_row['Hit@10 (%)']:.2f}%）。"
    )
    lines.append("")
    lines.append(
        "指標說明：Hit@K 是指在檢索結果的前 K 名之中，"
        "是否至少命中一個正確答案所在的段落或文件，數值愈高代表檢索系統愈容易在前 K 名內找到相關內容；"
        "MRR（Mean Reciprocal Rank，平均倒數排名）則是針對每個查詢，"
        "計算第一個命中結果所在排名的倒數（例如第一名命中則為 1，第二名命中則為 0.5），"
        "再取所有查詢的平均值，能反映正確結果平均排得多前面；"
        "File-level Recall 則是在檔案層級上，"
        "衡量前 K 名檢索結果總共涵蓋了多少比例應該被找到的正確證據檔案，"
        "著重於檔案（而非段落）是否被完整涵蓋。"
    )
    lines.append("")
    return "\n".join(lines)


def _cross_task_summary(qa_df, summarization_df, rag_df) -> str:
    """
    根據三個任務中實際存在的結果，產生簡短的交叉任務總結。

    參數：
        qa_df, summarization_df, rag_df: 各任務的 DataFrame，若該任務結果不存在則為 None

    回傳：
        總結文字字串
    """
    lines = ["## 交叉任務簡短總結", ""]

    available_tasks = []
    if qa_df is not None:
        available_tasks.append("QA")
    if summarization_df is not None:
        available_tasks.append("文本摘要")
    if rag_df is not None:
        available_tasks.append("RAG")

    if not available_tasks:
        lines.append("目前尚無任何任務的評估結果可供總結。")
        lines.append("")
        return "\n".join(lines)

    lines.append(
        f"本次分析涵蓋了 {'、'.join(available_tasks)} 共 {len(available_tasks)} 項任務的真實評估結果。"
        "整體而言，三項任務分別採用了不同性質的評估指標："
        "QA 任務著重於答案的精確度與部分匹配程度，"
        "文本摘要任務同時考量詞彙重疊與語義相似度，"
        "RAG 任務則聚焦於檢索排名品質與證據涵蓋率。"
    )
    lines.append(
        "由於不同任務使用的模型、資料集與指標並不相同，"
        "無法直接跨任務比較分數高低，"
        "但可以觀察到：在各自任務中，通常是針對該任務或相近領域微調過的模型"
        "（例如 QA 任務中專門在 SQuAD 上微調的模型），表現會優於未經微調的通用模型。"
    )
    lines.append("")
    return "\n".join(lines)


def generate_analysis_report() -> str:
    """
    產生完整的彙總分析報告（繁體中文字串）。

    依序讀取 QA、文本摘要、RAG 三個任務的真實結果 CSV，
    若某任務的 CSV 不存在，該段落會註明「結果尚未產生」，不會編造數字。

    回傳：
        完整分析報告字串
    """
    qa_df = _read_csv_if_exists(QA_CSV)
    summarization_df = _read_csv_if_exists(SUMMARIZATION_CSV)
    rag_df = _read_csv_if_exists(RAG_CSV)

    sections = [
        "# LLM 評估指標彙總分析報告",
        "",
        "本報告僅根據 results/ 目錄下實際存在的真實評估結果 CSV 檔案產生，",
        "未使用任何合成、模擬或假造的數據。",
        "",
        _analyze_qa(QA_CSV),
        _analyze_summarization(SUMMARIZATION_CSV),
        _analyze_rag(RAG_CSV),
        _cross_task_summary(qa_df, summarization_df, rag_df),
    ]

    return "\n".join(sections)


def main() -> None:
    """主程式進入點：產生分析報告，存檔並印出到終端機。"""
    report = generate_analysis_report()

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"分析報告已存至 {REPORT_PATH}")
    print("")
    print(report)


if __name__ == "__main__":
    main()
