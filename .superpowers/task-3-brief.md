# 任務 3：文本摘要模型評估

## 目標
實作文本摘要（Text Summarization）模型的評估流程。使用 CNN/DailyMail 數據集評估兩個預先訓練的摘要模型，計算 ROUGE 分數和 BERTScore 指標。

## 需要建立/修改的文件

### 1. src/metrics_utils.py（修改）
向現有的 metrics_utils.py 添加摘要指標計算函數：

**calculate_summarization_metrics(predictions, references)**
- 計算摘要評估指標
- 參數：
  - predictions：生成的摘要列表
  - references：參考摘要列表
- 需要使用 `evaluate` 套件中的 ROUGE 和 BERTScore
- 返回字典：
  ```python
  {
    'rouge1': XX.XX,      # ROUGE-1 分數（百分比）
    'rouge2': XX.XX,      # ROUGE-2 分數（百分比）
    'rougeL': XX.XX,      # ROUGE-L 分數（百分比）
    'bertscore': XX.XX    # BERTScore F1 平均值（百分比）
  }
  ```

實作步驟：
1. 使用 `evaluate.load('rouge')` 載入 ROUGE 評估器
2. 使用 `evaluate.load('bertscore')` 載入 BERTScore 評估器
3. 計算 ROUGE 分數（使用語幹提取）
4. 計算 BERTScore（lang="en"）
5. 將所有分數轉換為百分比格式

### 2. src/summarization_evaluation.py（建立）
建立摘要評估主腳本。需要實作以下函數和流程：

**load_summarization_dataset(dataset_name="cnn_dailymail", config="3.0.0", split="validation", sample_size=None)**
- 從 Hugging Face 載入 CNN/DailyMail 數據集
- 返回數據集物件

**prepare_summarization_data(dataset)**
- 從數據集中提取文章和參考摘要
- 返回兩個列表：(articles, references)

**evaluate_summarization_model(model_name, dataset, num_samples=None)**
- 評估單個摘要模型
- 參數：model_name、dataset、num_samples
- 使用 pipeline("summarization", model=model_name) 進行推理
- 對於超過 1024 字詞的文章進行截斷
- 生成摘要設定：max_length=150, min_length=50
- 返回字典包含：
  - model：模型名稱
  - rouge1、rouge2、rougeL、bertscore：各項指標分數
  - num_samples：評估樣本數

**main()**
- 主入口點
- 載入 config.json 獲取模型和數據集配置
- 為每個模型執行 evaluate_summarization_model
- 將結果保存為：
  - results/summarization_results.csv（列：Model, ROUGE-1 (%), ROUGE-2 (%), ROUGE-L (%), BERTScore (%), Num Samples）
  - results/summarization_results.json（詳細結果，包含時間戳）
- 打印結果表格到控制台

## 數據集說明
- **CNN/DailyMail v3.0.0**：新聞摘要數據集
- 驗證集約 13K 篇文章
- 格式：{article, highlights}
- 文章平均長度：數百字詞
- 摘要為多句摘要

## 模型配置（來自 config.json）
應評估以下兩個模型：
1. `facebook/bart-large-cnn`：BART 大型模型，在 CNN/DailyMail 上微調
2. `google/pegasus-cnn_dailymail`：PEGASUS 模型，在 CNN/DailyMail 上微調

## 指標說明
- **ROUGE-1**：單元語法（unigram）的精確度、召回率和 F1 分數
- **ROUGE-2**：二元語法（bigram）的精確度、召回率和 F1 分數
- **ROUGE-L**：最長公共子序列（LCS）的精確度、召回率和 F1 分數
- **BERTScore**：使用 BERT 內文本化詞嵌入計算的語義相似度

## 全域約束
- 所有註釋和文檔必須使用**繁體中文**
- 代碼變數和函數名保持英文
- 結果必須儲存為 CSV 和 JSON
- 為測試使用 50 個樣本（摘要評估較耗時）

## 驗收標準
1. 向 metrics_utils.py 成功添加了摘要指標函數
2. summarization_evaluation.py 成功載入 CNN/DailyMail 數據集
3. 兩個模型都成功評估（使用 50 個樣本）
4. 所有四個指標都已計算（ROUGE-1/2/L 和 BERTScore）
5. 結果文件格式正確
6. CSV 和 JSON 檔案都已生成

## 測試命令
```bash
python src/summarization_evaluation.py
```

應輸出：
- 進度條和狀態消息
- 包含所有指標的結果表格
- 確認文件已保存

## 回報格式
完成後，請在 `.superpowers/task-3-report.md` 中提交報告，包含：
1. 實作的函數概述
2. 測試結果（兩個模型的所有指標）
3. 任何問題或疑慮
4. 自審查發現
