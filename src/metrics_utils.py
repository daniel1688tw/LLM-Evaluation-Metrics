"""
共享指標計算工具

提供不同任務的指標計算函數，包括 QA、摘要化和 RAG 等。
"""

import re
import string
from typing import List, Dict, Union


def normalize_answer(s: str) -> str:
    """
    正規化答案文本以便進行比較。

    移除冠詞(a, an, the)、修正空白、移除標點符號、轉小寫。
    這是標準的 SQuAD 評估正規化方法。

    參數：
        s: 待正規化的答案文本

    返回：
        正規化後的答案字符串
    """
    # 轉換為小寫
    s = s.lower()

    # 移除冠詞
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    # 移除標點符號
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    # 修正空白（移除多余空格）
    def white_space_fix(text):
        return ' '.join(text.split())

    # 按順序應用各種正規化操作
    result = white_space_fix(remove_articles(remove_punc(s)))
    return result


def exact_match_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算精確匹配分數。

    比較預測和參考答案是否完全相同（在正規化後）。
    如果 ground_truth 是列表，檢查預測是否與列表中的任何一個相匹配
    （遵循 SQuAD 評估標準）。

    參數：
        prediction: 預測答案
        ground_truth: 參考答案或參考答案列表

    返回：
        分數（0 或 1）
    """
    # 正規化預測答案
    prediction_normalized = normalize_answer(prediction)

    # 如果 ground_truth 是列表，檢查是否與任何一個匹配
    if isinstance(ground_truth, list):
        ground_truth_list = ground_truth
    else:
        ground_truth_list = [ground_truth]

    # 檢查預測是否與任何參考答案完全相同（SQuAD 標準）
    for gt in ground_truth_list:
        gt_normalized = normalize_answer(gt)
        if prediction_normalized == gt_normalized:
            return 1.0

    return 0.0


def f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算令牌級別的 F1 分數。

    使用共同令牌計算精確度和召回率，然後計算 F1 分數。
    如果 ground_truth 是列表，取最高的 F1 分數（SQuAD 評估標準）。

    參數：
        prediction: 預測答案
        ground_truth: 參考答案或參考答案列表

    返回：
        F1 分數（0 到 1 之間）
    """
    # 正規化預測答案
    prediction_normalized = normalize_answer(prediction)
    prediction_tokens = set(prediction_normalized.split())

    # 如果 ground_truth 是列表，取最高的 F1 分數
    if isinstance(ground_truth, list):
        ground_truth_list = ground_truth
    else:
        ground_truth_list = [ground_truth]

    f1_scores = []
    for gt in ground_truth_list:
        gt_normalized = normalize_answer(gt)
        gt_tokens = set(gt_normalized.split())

        # 計算共同令牌
        common_tokens = prediction_tokens & gt_tokens

        # 如果沒有共同令牌，F1 分數為 0
        if len(common_tokens) == 0:
            f1_scores.append(0.0)
            continue

        # 計算精確度和召回率
        precision = len(common_tokens) / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
        recall = len(common_tokens) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0

        # 計算 F1 分數
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores.append(f1)

    # 返回最高 F1 分數（遵循 SQuAD 評估標準）
    if f1_scores:
        return max(f1_scores)
    return 0.0


def calculate_qa_metrics(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, float]:
    """
    計算 QA 指標的總體統計。

    計算整個預測集上的平均精確匹配率和 F1 分數。
    遵循 SQuAD 評估標準（對多個參考答案取最高分）。

    參數：
        predictions: 預測答案列表
        references: 參考答案列表（可以是列表的列表，對應多個有效答案）

    返回：
        字典，包含：
        - 'exact_match': 精確匹配百分比（0-100）
        - 'f1': F1 分數百分比（0-100）
    """
    if len(predictions) != len(references):
        raise ValueError(f"預測數量 ({len(predictions)}) 必須等於參考答案數量 ({len(references)})")

    if len(predictions) == 0:
        return {'exact_match': 0.0, 'f1': 0.0}

    # 計算每個樣本的分數
    exact_matches = []
    f1_scores = []

    for pred, ref in zip(predictions, references):
        exact_matches.append(exact_match_score(pred, ref))
        f1_scores.append(f1_score(pred, ref))

    # 計算平均分數並轉換為百分比
    avg_exact_match = (sum(exact_matches) / len(exact_matches)) * 100
    avg_f1 = (sum(f1_scores) / len(f1_scores)) * 100

    return {
        'exact_match': round(avg_exact_match, 2),
        'f1': round(avg_f1, 2)
    }


def calculate_summarization_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    計算文本摘要評估指標。

    使用 ROUGE 和 BERTScore 評估生成的摘要質量。
    ROUGE 衡量 n-gram 重疊，BERTScore 衡量語義相似度。

    參數：
        predictions: 生成的摘要列表
        references: 參考摘要列表

    返回：
        字典，包含：
        - 'rouge1': ROUGE-1 F1 分數（百分比）
        - 'rouge2': ROUGE-2 F1 分數（百分比）
        - 'rougeL': ROUGE-L F1 分數（百分比）
        - 'bertscore': BERTScore F1 平均值（百分比）
    """
    if len(predictions) != len(references):
        raise ValueError(f"預測數量 ({len(predictions)}) 必須等於參考摘要數量 ({len(references)})")

    if len(predictions) == 0:
        return {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0,
            'bertscore': 0.0
        }

    try:
        from rouge_score import rouge_scorer
    except ImportError:
        raise ImportError("請安裝 rouge_score 套件：pip install rouge_score")

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        raise ImportError("請安裝 bert_score 套件：pip install bert_score")

    # 計算 ROUGE 分數
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    # 計算平均 ROUGE 分數
    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0

    # 計算 BERTScore
    try:
        precision, recall, f1 = bert_score_fn(
            predictions,
            references,
            lang="en",
            batch_size=32,
            device=None  # 自動選擇 GPU 或 CPU
        )
        avg_bertscore = f1.mean().item()
    except Exception as e:
        # 如果 BERTScore 計算失敗（例如缺少模型），返回 0
        avg_bertscore = 0.0

    # 返回所有指標（轉換為百分比格式）
    return {
        'rouge1': round(avg_rouge1 * 100, 2),
        'rouge2': round(avg_rouge2 * 100, 2),
        'rougeL': round(avg_rougeL * 100, 2),
        'bertscore': round(avg_bertscore * 100, 2)
    }
