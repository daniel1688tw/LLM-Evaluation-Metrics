"""
RAG 檢索評估指標工具

提供 file-level（檔案層級）檢索評估指標的計算：
- Hit@K：前 K 個檢索結果中是否至少命中一個正確證據檔案
- MRR：第一個正確證據檔案的倒數排名平均
- file-level Recall：前 K 個結果涵蓋了多少比例的正確證據檔案

所有函數以「排序後的檢索檔案 id 列表」與「正確證據檔案 id 集合」為輸入。
"""

from typing import List, Set, Dict
import numpy as np


def hit_at_k(ranked_files: List[str], ground_truth: Set[str], k: int) -> float:
    """前 K 個檢索結果中若含任一正確證據檔案則為 1.0，否則 0.0。"""
    if not ground_truth:
        return 0.0
    return 1.0 if any(f in ground_truth for f in ranked_files[:k]) else 0.0


def reciprocal_rank(ranked_files: List[str], ground_truth: Set[str]) -> float:
    """回傳第一個正確證據檔案的倒數排名（第 1 名為 1.0，第 2 名為 0.5…）；找不到為 0。"""
    if not ground_truth:
        return 0.0
    for rank, f in enumerate(ranked_files, start=1):
        if f in ground_truth:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_files: List[str], ground_truth: Set[str], k: int) -> float:
    """file-level Recall@K：前 K 個結果涵蓋的正確證據檔案數 / 正確證據檔案總數。"""
    if not ground_truth:
        return 0.0
    retrieved = set(ranked_files[:k])
    return len(retrieved & ground_truth) / len(ground_truth)


def compute_rag_metrics(
    all_ranked: List[List[str]],
    all_ground_truth: List[Set[str]],
    recall_k: int = 10,
) -> Dict[str, float]:
    """
    彙總所有查詢的 RAG 檢索指標。

    參數：
        all_ranked: 每個查詢的「排序後檢索檔案 id 列表」
        all_ground_truth: 每個查詢的「正確證據檔案 id 集合」
        recall_k: file-level Recall 的截斷位置（預設 10）

    回傳：
        dict，含 hit@1、hit@5、hit@10、mrr、recall（皆為百分比 0-100）
    """
    hits1, hits5, hits10, mrrs, recalls = [], [], [], [], []
    for ranked, gt in zip(all_ranked, all_ground_truth):
        hits1.append(hit_at_k(ranked, gt, 1))
        hits5.append(hit_at_k(ranked, gt, 5))
        hits10.append(hit_at_k(ranked, gt, 10))
        mrrs.append(reciprocal_rank(ranked, gt))
        recalls.append(recall_at_k(ranked, gt, recall_k))

    return {
        "hit@1": float(np.mean(hits1)) * 100,
        "hit@5": float(np.mean(hits5)) * 100,
        "hit@10": float(np.mean(hits10)) * 100,
        "mrr": float(np.mean(mrrs)) * 100,
        "recall": float(np.mean(recalls)) * 100,
    }
