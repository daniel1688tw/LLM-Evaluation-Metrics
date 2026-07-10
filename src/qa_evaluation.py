"""
QA 模型評估腳本

使用 SQuAD 數據集評估問答模型。
計算精確匹配率和 F1 分數指標。
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import torch

from metrics_utils import calculate_qa_metrics


def load_qa_dataset(
    dataset_name: str = "squad",
    split: str = "validation",
    sample_size: Optional[int] = None
) -> Dict:
    """
    從 Hugging Face 載入 QA 數據集（SQuAD）。

    參數：
        dataset_name: 數據集名稱（默認："squad"）
        split: 分割方式（默認："validation"）
        sample_size: 樣本限制（用於測試），None 表示使用全部數據

    返回：
        加載的數據集對象
    """
    print(f"正在載入 {dataset_name} 數據集的 {split} 分割...")
    dataset = load_dataset(dataset_name, split=split)

    # 如果指定了樣本大小，則進行截取
    if sample_size is not None:
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        print(f"已限制為 {len(dataset)} 個樣本用於測試")
    else:
        print(f"已加載 {len(dataset)} 個樣本")

    return dataset


def prepare_qa_data(dataset) -> Tuple[List[str], List[str], List[List[str]]]:
    """
    從數據集中提取問題、上下文、參考答案。

    參數：
        dataset: 加載的數據集

    返回：
        三個列表的元組：(questions, contexts, references)
        其中 references 是答案文本列表的列表
    """
    questions = []
    contexts = []
    references = []

    print("正在準備 QA 數據...")
    for sample in tqdm(dataset, desc="準備數據"):
        question = sample.get("question", "")
        context = sample.get("context", "")

        # 提取所有參考答案
        answers = sample.get("answers", {})
        answer_texts = answers.get("text", []) if isinstance(answers, dict) else []

        questions.append(question)
        contexts.append(context)
        # 如果沒有答案，使用空字符串列表
        references.append(answer_texts if answer_texts else [""])

    return questions, contexts, references


def evaluate_qa_model(
    model_name: str,
    questions: List[str],
    contexts: List[str],
    references: List[List[str]],
    num_samples: Optional[int] = None
) -> Dict:
    """
    評估單個 QA 模型。

    參數：
        model_name: Hugging Face 模型標識
        questions: 問題列表
        contexts: 上下文列表
        references: 參考答案列表（列表的列表）
        num_samples: 評估樣本數（None 表示使用全部）

    返回：
        字典包含：
        - 'model': 模型名稱
        - 'exact_match': 準確度百分比
        - 'f1_score': F1 分數百分比
        - 'num_samples': 評估樣本數
        - 'timestamp': 評估時間戳
    """
    # 限制樣本數
    if num_samples is not None:
        questions = questions[:num_samples]
        contexts = contexts[:num_samples]
        references = references[:num_samples]

    actual_samples = len(questions)

    print(f"\n開始評估模型：{model_name}")
    print(f"評估樣本數：{actual_samples}")

    # 加載模型和分詞器
    print(f"正在加載模型 {model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        print(f"模型已加載到 {device} 設備")
    except Exception as e:
        print(f"錯誤：無法加載模型 {model_name}: {str(e)}")
        raise

    # 進行模型推理
    predictions = []
    print(f"正在對 {model_name} 進行推理...")

    for question, context in tqdm(zip(questions, contexts), total=len(questions), desc="推理進行中"):
        try:
            # 準備輸入
            inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)

            # 執行推理
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 解析答案
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            # 找到最可能的開始和結束位置
            answer_start_idx = torch.argmax(start_logits)
            answer_end_idx = torch.argmax(end_logits)

            # 確保結束位置在開始位置之後
            if answer_start_idx <= answer_end_idx:
                answer_ids = input_ids[0, answer_start_idx:answer_end_idx + 1]
            else:
                answer_ids = input_ids[0, answer_start_idx:answer_start_idx + 1]

            # 解碼答案
            prediction = tokenizer.decode(answer_ids, skip_special_tokens=True)

            predictions.append(prediction)
        except Exception as e:
            print(f"錯誤：推理失敗，問題：{question[:50]}... 錯誤：{str(e)}")
            predictions.append("")

    # 計算指標
    print(f"正在計算 {model_name} 的指標...")
    metrics = calculate_qa_metrics(predictions, references)

    # 返回結果
    return {
        'model': model_name,
        'exact_match': metrics['exact_match'],
        'f1_score': metrics['f1'],
        'num_samples': actual_samples,
        'timestamp': datetime.now().isoformat()
    }


def save_results(results: List[Dict], output_dir: str = "results") -> None:
    """
    將評估結果保存為 CSV 和 JSON 格式。

    參數：
        results: 包含評估結果的字典列表
        output_dir: 輸出目錄（默認："results"）
    """
    # 建立輸出目錄（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 轉換為 DataFrame 用於 CSV 保存
    results_for_csv = []
    for result in results:
        results_for_csv.append({
            'Model': result['model'],
            'Exact Match (%)': f"{result['exact_match']:.2f}",
            'F1 Score (%)': f"{result['f1_score']:.2f}",
            'Num Samples': result['num_samples']
        })

    df = pd.DataFrame(results_for_csv)

    # 保存為 CSV
    csv_path = os.path.join(output_dir, 'qa_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"已將結果保存到 {csv_path}")

    # 保存為 JSON
    json_path = os.path.join(output_dir, 'qa_results.json')
    json_results = {
        'timestamp': datetime.now().isoformat(),
        'task': 'qa_evaluation',
        'dataset': 'squad',
        'results': results
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"已將結果保存到 {json_path}")

    # 打印結果表格到控制台
    print("\n" + "="*70)
    print("QA 模型評估結果")
    print("="*70)
    print(df.to_string(index=False))
    print("="*70)


def main():
    """
    主入口點。

    加載配置、數據集和模型，執行評估，保存結果。
    """
    # 加載配置
    config_path = "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    qa_config = config.get('qa', {})
    dataset_config = qa_config.get('dataset', {})
    models = qa_config.get('models', [])

    # 從配置中提取數據集參數
    dataset_name = dataset_config.get('name', 'squad')
    split = dataset_config.get('split', 'validation')

    # 測試時使用 100 個樣本，加快開發速度
    sample_size = 100

    # 加載數據集
    dataset = load_qa_dataset(
        dataset_name=dataset_name,
        split=split,
        sample_size=sample_size
    )

    # 準備數據
    questions, contexts, references = prepare_qa_data(dataset)

    # 評估每個模型
    results = []
    for model_name in models:
        try:
            result = evaluate_qa_model(
                model_name=model_name,
                questions=questions,
                contexts=contexts,
                references=references,
                num_samples=None  # 使用準備好的數據中的全部
            )
            results.append(result)
        except Exception as e:
            print(f"評估模型 {model_name} 時發生錯誤：{str(e)}")
            continue

    # 保存結果
    if results:
        save_results(results)
        print("\n評估完成！")
    else:
        print("未成功評估任何模型")


if __name__ == "__main__":
    main()
