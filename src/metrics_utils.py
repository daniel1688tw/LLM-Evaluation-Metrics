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
    如果 ground_truth 是列表，檢查預測是否與列表中的任何一個相匹配。

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

    # 檢查預測是否與任何參考答案完全相同
    for gt in ground_truth_list:
        gt_normalized = normalize_answer(gt)
        if prediction_normalized == gt_normalized:
            return 1.0

    return 0.0


def f1_score(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    計算令牌級別的 F1 分數。

    使用共同令牌計算精確度和召回率，然後計算 F1 分數。
    如果 ground_truth 是列表，計算與列表中所有答案的平均 F1 分數。

    參數：
        prediction: 預測答案
        ground_truth: 參考答案或參考答案列表

    返回：
        F1 分數（0 到 1 之間）
    """
    # 正規化預測答案
    prediction_normalized = normalize_answer(prediction)
    prediction_tokens = set(prediction_normalized.split())

    # 如果 ground_truth 是列表，計算與列表中所有答案的平均 F1 分數
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

    # 返回平均 F1 分數
    if f1_scores:
        return sum(f1_scores) / len(f1_scores)
    return 0.0


def calculate_qa_metrics(
    predictions: List[str],
    references: List[Union[str, List[str]]]
) -> Dict[str, float]:
    """
    計算 QA 指標的總體統計。

    計算整個預測集上的平均精確匹配率和 F1 分數。

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
