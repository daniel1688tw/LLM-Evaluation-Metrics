"""
RAG 檢索評估腳本（HippoCamp 基準）

任務：在 HippoCamp 基準的 factual_retention 查詢上，評估 retriever（句向量）模型
      的 file-level（檔案層級）檢索能力。

作法：
  1. 從 Hugging Face 下載 HippoCamp 的 viewer_parquet（含 query 與正確證據檔案）。
  2. 以「所有查詢證據檔案的聯集」建立檢索語料（每個檔案一段解析文字）。
  3. 對每個 retriever 模型，將語料與查詢編碼成句向量，用餘弦相似度排序檔案。
  4. 計算 Hit@1/5/10、MRR、file-level Recall。

註：作業原文於 RAG 寫「summarization models」為上一題複製貼上的筆誤；檢索指標
    （Hit@K/MRR/Recall）需要的是 retriever/embedding 模型，故此處改用句向量模型。
"""

import json
import os
from datetime import datetime
from typing import List, Set, Dict, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer
import torch

from rag_utils import compute_rag_metrics

# 語料每個檔案取用的最大字元數（句向量模型本身也會截斷 token，此處先行限制記憶體用量）
MAX_DOC_CHARS = 4000
# BGE 系列模型建議在查詢前加入檢索指令以提升效果
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _to_str_list(value) -> List[str]:
    """將 parquet 中的 list/ndarray/None 欄位安全轉為 str 列表。"""
    if value is None:
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        return [str(v) for v in value]
    return [str(value)]


def load_hippocamp(
    repo_id: str, users: List[str], split: str, qa_type: str
) -> Tuple[Dict[str, str], List[Dict]]:
    """
    載入 HippoCamp 資料，建立檢索語料與查詢清單。

    回傳：
        corpus: {file_path: 文字內容} 的字典（檢索候選檔案）
        queries: [{'id', 'question', 'ground_truth': set(file_path)} , ...]
    """
    corpus: Dict[str, str] = {}
    queries: List[Dict] = []

    for user in users:
        filename = f"viewer_parquet/{user}_{split}/{qa_type}.parquet"
        print(f"下載 {repo_id} :: {filename} ...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        df = pd.read_parquet(path)

        for _, row in df.iterrows():
            file_paths = _to_str_list(row.get("file_path"))
            file_texts = _to_str_list(row.get("file_text"))

            # 將此查詢的證據檔案（與其文字）加入語料
            for i, fp in enumerate(file_paths):
                if fp not in corpus:
                    text = file_texts[i] if i < len(file_texts) else ""
                    corpus[fp] = (text or "")[:MAX_DOC_CHARS]

            gt = set(file_paths)
            question = row.get("question")
            if question and gt:
                queries.append(
                    {
                        "id": f"{user}-{row.get('id')}",
                        "question": str(question),
                        "ground_truth": gt,
                    }
                )

    print(f"語料檔案數：{len(corpus)}；查詢數：{len(queries)}")
    return corpus, queries


def evaluate_retriever(
    model_name: str, corpus: Dict[str, str], queries: List[Dict]
) -> Dict:
    """以單一 retriever 模型執行檢索並計算指標。"""
    print(f"\n評估 retriever 模型：{model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用裝置：{device}")

    model = SentenceTransformer(model_name, device=device)

    file_ids = list(corpus.keys())
    doc_texts = [corpus[f] for f in file_ids]

    is_bge = "bge" in model_name.lower()
    query_texts = [
        (BGE_QUERY_PREFIX + q["question"]) if is_bge else q["question"] for q in queries
    ]

    print(f"編碼語料（{len(doc_texts)} 個檔案）...")
    doc_emb = model.encode(
        doc_texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )
    print(f"編碼查詢（{len(query_texts)} 個）...")
    query_emb = model.encode(
        query_texts, batch_size=32, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True,
    )

    # 已正規化 → 內積即餘弦相似度。對每個查詢取相似度最高的檔案排序。
    print("計算相似度與排序...")
    all_ranked: List[List[str]] = []
    all_gt: List[Set[str]] = []
    for i in tqdm(range(len(queries)), desc="檢索"):
        sims = doc_emb @ query_emb[i]
        order = np.argsort(-sims)
        all_ranked.append([file_ids[j] for j in order])
        all_gt.append(queries[i]["ground_truth"])

    metrics = compute_rag_metrics(all_ranked, all_gt, recall_k=10)
    return {
        "model": model_name,
        "hit@1": metrics["hit@1"],
        "hit@5": metrics["hit@5"],
        "hit@10": metrics["hit@10"],
        "mrr": metrics["mrr"],
        "recall": metrics["recall"],
        "num_queries": len(queries),
        "corpus_size": len(corpus),
        "timestamp": datetime.now().isoformat(),
    }


def save_results(results: List[Dict], output_dir: str = "results") -> None:
    """將結果存成 CSV 與 JSON 並印出表格。"""
    os.makedirs(output_dir, exist_ok=True)

    rows = [
        {
            "Model": r["model"],
            "Hit@1 (%)": f"{r['hit@1']:.2f}",
            "Hit@5 (%)": f"{r['hit@5']:.2f}",
            "Hit@10 (%)": f"{r['hit@10']:.2f}",
            "MRR (%)": f"{r['mrr']:.2f}",
            "File-level Recall@10 (%)": f"{r['recall']:.2f}",
            "Num Queries": r["num_queries"],
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "rag_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"已將結果保存到 {csv_path}")

    json_path = os.path.join(output_dir, "rag_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "task": "rag_evaluation",
                "dataset": "MMMem-org/HippoCamp (factual_retention)",
                "results": results,
            },
            f, indent=2, ensure_ascii=False,
        )
    print(f"已將結果保存到 {json_path}")

    print("\n" + "=" * 100)
    print("RAG 檢索評估結果（HippoCamp file-level 檢索）")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


def main():
    with open("config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    rag_cfg = config["rag"]
    ds = rag_cfg["dataset"]

    corpus, queries = load_hippocamp(
        repo_id=ds["name"],
        users=ds["users"],
        split=ds["split"],
        qa_type=ds["qa_type"],
    )

    results = []
    for model_name in rag_cfg["models"]:
        results.append(evaluate_retriever(model_name, corpus, queries))

    save_results(results)
    print("\n評估完成！")


if __name__ == "__main__":
    main()
