# 任務 4：RAG 模型評估

## 目標
實作檢索增強生成（RAG）的評估流程。使用 HippoCamp 基準（或模擬版本）評估兩個模型，計算 Hit@K、MRR 和召回率指標。

## 需要建立的文件

### 1. src/rag_utils.py（建立）
建立 RAG 特定的指標計算工具。需要實作以下函數：

**calculate_hit_at_k(retrieved_indices, ground_truth_indices, k)**
- 計算 Hit@K 指標
- 參數：
  - retrieved_indices：按相關性排序的檢索文檔索引列表
  - ground_truth_indices：相關文檔索引列表
  - k：計算 Hit 的截止位置
- 返回：1.0（如果前 K 個結果中有相關文檔）或 0.0

**calculate_mrr(retrieved_indices, ground_truth_indices)**
- 計算平均倒數排名（Mean Reciprocal Rank）
- 返回第一個相關文檔的倒數排名，或 0.0（如果未找到）

**calculate_recall(retrieved_set, ground_truth_set)**
- 計算召回率：檢索到的相關文檔數 / 總相關文檔數
- 參數可以是集合（set）
- 返回 0 到 1 之間的召回率分數

**calculate_rag_metrics(all_retrieved, all_ground_truth)**
- 計算所有 RAG 指標的統計
- 參數：
  - all_retrieved：每個查詢的檢索索引列表
  - all_ground_truth：每個查詢的相關索引列表
- 返回字典：
  ```python
  {
    'hit@1': XX.XX,    # Hit@1 百分比
    'hit@5': XX.XX,    # Hit@5 百分比
    'hit@10': XX.XX,   # Hit@10 百分比
    'mrr': XX.XX,      # MRR 百分比
    'recall': XX.XX    # 召回率百分比
  }
  ```

### 2. src/rag_evaluation.py（建立）
建立 RAG 評估主腳本。需要實作以下函數和流程：

**create_mock_rag_dataset(num_queries=100)**
- 創建模擬 RAG 數據集（模擬 HippoCamp 結構）
- 生成 1000 個合成文檔
- 為每個查詢生成相關文檔的地面真實標籤
- 返回字典：{queries: [...], documents: [...], ground_truth: [...]}

**embed_and_retrieve(query, documents, num_retrieve=50)**
- 模擬文檔檢索
- 基於查詢字詞與文檔字詞的重疊評分文檔
- 按相關性分數排序
- 返回前 num_retrieve 個文檔的索引列表

**evaluate_rag_model(model_name, dataset, num_samples=None)**
- 評估 RAG 模型性能
- 參數：model_name、dataset、num_samples
- 針對每個查詢執行文檔檢索
- 計算 RAG 指標
- 返回字典包含：
  - model：模型名稱
  - hit@1、hit@5、hit@10、mrr、recall：各項指標
  - num_queries：評估的查詢數

**main()**
- 主入口點
- 載入 config.json 獲取模型配置
- 創建模擬 RAG 數據集（100 個查詢）
- 為每個模型執行評估
- 將結果保存為：
  - results/rag_results.csv（列：Model, Hit@1 (%), Hit@5 (%), Hit@10 (%), MRR (%), Recall (%), Num Queries）
  - results/rag_results.json（詳細結果）
- 打印結果表格到控制台

## RAG 評估說明

### 數據集結構
- **查詢**：使用者搜尋查詢
- **文檔集合**：可檢索的候選文檔
- **地面真實**：每個查詢的相關文檔列表

### 評估指標
- **Hit@K**：在前 K 個結果中至少有一個相關文檔的查詢百分比
- **MRR（Mean Reciprocal Rank）**：第一個相關結果的排名的倒數的平均值
  - 完美排名（第 1）= 1.0
  - 排名在第 2 = 0.5
  - 未找到 = 0.0
- **召回率（Recall）**：檢索集合中相關文檔佔所有相關文檔的比例

### 模型配置（來自 config.json）
評估模型：
1. `facebook/bart-large-cnn`
2. `google/pegasus-cnn_dailymail`

注意：在 RAG 評估中，模型用於文檔相關性評分/排名，而不是生成摘要

## 全域約束
- 所有註釋和文檔必須使用**繁體中文**
- 代碼變數和函數名保持英文
- 結果必須儲存為 CSV 和 JSON
- 使用 numpy 進行統計計算
- 使用 tqdm 顯示進度條

## 驗收標準
1. rag_utils.py 實作了所有指標計算函數
2. rag_evaluation.py 成功創建模擬 RAG 數據集
3. 兩個模型都成功評估（100 個查詢）
4. 所有五個指標都已計算
5. 結果文件格式正確
6. CSV 和 JSON 檔案都已生成

## 測試命令
```bash
python src/rag_evaluation.py
```

應輸出：
- 數據集創建消息
- 進度條和檢索狀態
- 包含所有指標的結果表格
- 確認文件已保存

## 回報格式
完成後，請在 `.superpowers/task-4-report.md` 中提交報告，包含：
1. 實作的函數概述
2. 模擬數據集說明
3. 測試結果（兩個模型的所有指標）
4. 任何問題或疑慮
5. 自審查發現
