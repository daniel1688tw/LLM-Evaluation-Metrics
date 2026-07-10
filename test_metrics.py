"""
測試指標計算函數

驗證 metrics_utils.py 中的函數是否正常工作。
"""

import sys
sys.path.insert(0, 'src')

from metrics_utils import normalize_answer, exact_match_score, f1_score, calculate_qa_metrics


def test_normalize_answer():
    """測試答案正規化函數"""
    print("測試 normalize_answer()...")

    # 測試冠詞移除
    assert normalize_answer("The capital of France") == "capital of france"
    assert normalize_answer("a dog") == "dog"
    assert normalize_answer("an apple") == "apple"

    # 測試標點符號移除
    assert normalize_answer("Hello, world!") == "hello world"
    assert normalize_answer("What?") == "what"

    # 測試空白修正
    assert normalize_answer("  multiple   spaces  ") == "multiple spaces"

    print("[OK] normalize_answer() 測試通過")


def test_exact_match_score():
    """測試精確匹配分數函數"""
    print("測試 exact_match_score()...")

    # 完全匹配（忽略標點和大小寫）
    assert exact_match_score("The answer is yes", "the answer is yes") == 1.0
    assert exact_match_score("Paris", "paris") == 1.0

    # 不匹配
    assert exact_match_score("yes", "no") == 0.0

    # 與多個參考答案的匹配
    assert exact_match_score("apple", ["orange", "apple", "banana"]) == 1.0
    assert exact_match_score("grape", ["orange", "apple", "banana"]) == 0.0

    print("[OK] exact_match_score() 測試通過")


def test_f1_score():
    """測試 F1 分數函數"""
    print("測試 f1_score()...")

    # 完美匹配
    assert f1_score("The quick brown fox", "the quick brown fox") == 1.0

    # 部分匹配
    score = f1_score("The quick brown", "the quick brown fox")
    assert 0 < score < 1  # 應該在 0 和 1 之間

    # 完全不匹配
    assert f1_score("apple", "orange") == 0.0

    # 與多個參考答案的匹配
    score = f1_score("cat", ["dog", "cat", "bird"])
    assert score == 1.0  # 應該與其中一個完全匹配

    print("[OK] f1_score() 測試通過")


def test_calculate_qa_metrics():
    """測試整體 QA 指標計算函數"""
    print("測試 calculate_qa_metrics()...")

    # 簡單測試
    predictions = ["Paris", "Berlin", "Rome"]
    references = [["Paris"], ["Berlin"], ["Rome"]]

    metrics = calculate_qa_metrics(predictions, references)

    # 應該有完美匹配
    assert metrics['exact_match'] == 100.0
    assert metrics['f1'] == 100.0

    print("[OK] calculate_qa_metrics() 測試通過")

    # 測試部分匹配
    predictions = ["Paris is nice", "Berlin Wall", "Rome history"]
    references = [["Paris"], ["Berlin"], ["Rome"]]

    metrics = calculate_qa_metrics(predictions, references)

    # 應該有部分匹配
    assert 0 <= metrics['exact_match'] <= 100
    assert 0 <= metrics['f1'] <= 100

    print("[OK] calculate_qa_metrics() 部分匹配測試通過")


def main():
    """執行所有測試"""
    print("=" * 60)
    print("開始執行 metrics_utils.py 測試")
    print("=" * 60)

    try:
        test_normalize_answer()
        test_exact_match_score()
        test_f1_score()
        test_calculate_qa_metrics()

        print("\n" + "=" * 60)
        print("所有測試通過！")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n測試失敗：{str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\n發生錯誤：{str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
