# 任務 2：QA 模型評估

## 目標
實作問答（QA）模型的評估流程。使用 SQuAD v2 數據集評估兩個預先訓練的 QA 模型，計算 Exact Match（準確度）和 F1 Score 指標。

## 需要建立/修改的文件

### 1. src/__init__.py（建立）
建立模塊初始化文件。包含基本的套件描述。

### 2. src/metrics_utils.py（建立）
建立共享指標計算工具。需要實作以下函數：

**normalize_answer(s)**
- 正規化答案文本：移除冠詞(a, an, the)、修正空白、移除標點符號、轉小寫
- 用於準確比較不同格式的答案

**exact_match_score(prediction, ground_truth)**
- 比較預測和參考答案是否完全相同（在正規化後）
- 返回 0（不匹配）或 1（匹配）

**f1_score(prediction, ground_truth)**
- 計算令牌級別的 F1 分數
- 使用共同令牌計算精確度和召回率
- 返回 0 到 1 之間的分數

**calculate_qa_metrics(predictions, references)**
- 計算 QA 指標的總體統計
- 參數：
  - predictions：預測答案列表
  - references：參考答案列表（可以是列表的列表，對應多個有效答案）
- 返回字典：{'exact_match': XX.XX, 'f1': XX.XX}（百分比形式）

### 3. src/qa_evaluation.py（建立）
建立 QA 評估主腳本。需要實作以下函數和流程：

**load_qa_dataset(dataset_name="squad", split="validation", sample_size=None)**
- 從 Hugging Face 載入 SQuAD 數據集
- 參數：dataset_name（數據集名稱）、split（分割方式）、sample_size（樣本限制，用於測試）

**prepare_qa_data(dataset)**
- 從數據集中提取問題、上下文、參考答案
- 返回三個列表：(questions, contexts, references)

**evaluate_qa_model(model_name, dataset, num_samples=None)**
- 評估單個 QA 模型
- 參數：model_name（Hugging Face 模型標識）、dataset、num_samples
- 使用 pipeline("question-answering", model=model_name) 進行推理
- 返回字典包含：
  - model：模型名稱
  - exact_match：準確度百分比
  - f1_score：F1 分數百分比
  - num_samples：評估樣本數

**main()**
- 主入口點
- 載入 config.json 獲取模型和數據集配置
- 為每個模型執行 evaluate_qa_model
- 將結果保存為：
  - results/qa_results.csv（表格格式，列：Model, Exact Match (%), F1 Score (%), Num Samples）
  - results/qa_results.json（詳細結果，包含時間戳和每個模型的詳細分數）
- 打印結果表格到控制台

## 數據集說明
- **SQuAD v2.0**：斯坦福問答數據集第 2 版
- 驗證集約 12K 個問題
- 包含約 50% 無法回答的問題
- 格式：{question, context, answers{text[], answer_start}}

## 模型配置（來自 config.json）
應評估以下兩個模型：
1. `deepset/roberta-base-squad2`：RoBERTa 基礎模型，在 SQuAD 2.0 上微調
2. `bert-large-uncased-whole-word-masking-finetuned-squad`：BERT 大型模型，在 SQuAD 上微調

## 全域約束
- 所有註釋和文檔必須使用**繁體中文**
- 代碼變數和函數名保持英文
- 結果必須儲存為 CSV 和 JSON
- 使用 pandas 處理表格數據
- 使用 tqdm 顯示進度條
- 為測試使用 100 個樣本（加快開發速度）

## 驗收標準
1. metrics_utils.py 實作了所有四個函數
2. qa_evaluation.py 成功載入 SQuAD 數據集
3. 兩個模型都成功評估（使用 100 個樣本）
4. 結果文件格式正確，可以被讀取
5. CSV 和 JSON 檔案都已生成
6. 控制台輸出顯示清楚的結果表格

## 測試命令
```bash
python src/qa_evaluation.py
```

應輸出：
- 進度條和狀態消息
- 包含模型名稱、Exact Match 和 F1 Score 的結果表格
- 確認文件已保存到 results/ 目錄

## 回報格式
完成後，請在 `.superpowers/task-2-report.md` 中提交報告，包含：
1. 實作的函數概述
2. 測試結果（兩個模型的分數）
3. 任何問題或疑慮
4. 自審查發現
