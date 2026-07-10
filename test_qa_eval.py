"""
測試 QA 評估流程

驗證完整的 QA 評估流程是否正常工作。
"""

import sys
sys.path.insert(0, 'src')

from qa_evaluation import load_qa_dataset, prepare_qa_data, evaluate_qa_model


def main():
    """測試 QA 評估流程"""
    print("=" * 60)
    print("測試 QA 評估流程")
    print("=" * 60)

    # 加載數據集（限制為 10 個樣本用於快速測試）
    print("\n1. 測試數據集加載...")
    try:
        dataset = load_qa_dataset(
            dataset_name="squad",
            split="validation",
            sample_size=10
        )
        print(f"[OK] 數據集加載成功，共 {len(dataset)} 個樣本")
    except Exception as e:
        print(f"[ERROR] 數據集加載失敗：{str(e)}")
        return

    # 準備數據
    print("\n2. 測試數據準備...")
    try:
        questions, contexts, references = prepare_qa_data(dataset)
        print(f"[OK] 數據準備完成")
        print(f"    問題數：{len(questions)}")
        print(f"    上下文數：{len(contexts)}")
        print(f"    參考答案數：{len(references)}")

        # 顯示第一個樣本
        print(f"\n    第一個樣本：")
        print(f"    問題：{questions[0][:80]}")
        print(f"    參考答案：{references[0]}")
    except Exception as e:
        print(f"[ERROR] 數據準備失敗：{str(e)}")
        return

    # 測試模型評估（使用較小的模型）
    print("\n3. 測試模型評估...")
    print("    注意：這可能需要一段時間（首次需要下載模型）")

    model_name = "deepset/roberta-base-squad2"
    try:
        result = evaluate_qa_model(
            model_name=model_name,
            questions=questions,
            contexts=contexts,
            references=references,
            num_samples=5  # 只測試 5 個樣本
        )

        print(f"\n[OK] 模型評估完成")
        print(f"    模型：{result['model']}")
        print(f"    Exact Match：{result['exact_match']:.2f}%")
        print(f"    F1 分數：{result['f1_score']:.2f}%")
        print(f"    樣本數：{result['num_samples']}")
        print(f"    時間戳：{result['timestamp']}")
    except Exception as e:
        print(f"[ERROR] 模型評估失敗：{str(e)}")
        print("\n注意：這可能是由於 Transformers/PyTorch 版本不相容造成的。")
        return

    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
