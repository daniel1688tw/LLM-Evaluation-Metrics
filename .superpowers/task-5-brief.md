# 任務 5：結果視覺化與分析

## 目標
實作結果視覺化和分析報告生成。為三個評估任務的結果創建對比圖表，並生成汇總分析報告。

## 需要建立的文件

### 1. src/visualization.py（建立）
建立結果視覺化腳本。需要實作以下函數：

**plot_qa_results(results_csv='results/qa_results.csv')**
- 為 QA 結果創建對比圖表
- 使用 matplotlib 創建 1×2 的子圖
- 第一個子圖：模型的準確度（Exact Match）柱狀圖
- 第二個子圖：模型的 F1 分數柱狀圖
- 返回 matplotlib figure 物件

**plot_summarization_results(results_csv='results/summarization_results.csv')**
- 為摘要結果創建對比圖表
- 使用 matplotlib 創建 2×2 的子圖
- 四個子圖分別顯示：ROUGE-1、ROUGE-2、ROUGE-L、BERTScore
- 每個子圖都是模型對比的柱狀圖
- 返回 matplotlib figure 物件

**plot_rag_results(results_csv='results/rag_results.csv')**
- 為 RAG 結果創建對比圖表
- 使用 matplotlib 創建 2×3 的子圖
- 五個子圖分別顯示：Hit@1、Hit@5、Hit@10、MRR、Recall
- 返回 matplotlib figure 物件

**save_all_plots()**
- 生成所有三個對比圖表
- 檢查結果 CSV 文件是否存在
- 分別保存為：
  - results/qa_comparison.png
  - results/summarization_comparison.png
  - results/rag_comparison.png
- 使用 DPI 300，確保高質量

**main()**
- 調用 save_all_plots() 生成所有視覺化
- 打印確認消息

## 可視化要求
- 使用 pandas 讀取 CSV 文件
- 使用 matplotlib 創建圖表
- 所有圖表應該：
  - 有清晰的標題（繁體中文）
  - 有軸標籤
  - 有適當的顏色區分
  - 包含模型名稱在 x 軸上
  - 包含指標值在 y 軸上
- 圖表應該美觀、易於閱讀

### 2. src/analysis.py（建立）
建立分析報告生成腳本。需要實作以下函數：

**generate_analysis_report()**
- 生成汉总分析報告（繁體中文）
- 從 results/ 目錄讀取所有 CSV 文件
- 為每個任務生成分析部分：
  - **QA 分析**：
    - 打印 QA 結果表格
    - 報告最佳準確度和 F1 分數的模型
    - 解釋兩個指標的含義
  - **摘要分析**：
    - 打印摘要結果表格
    - 報告最佳 ROUGE-1 的模型
    - 解釋 ROUGE 和 BERTScore 指標的含義
  - **RAG 分析**：
    - 打印 RAG 結果表格
    - 報告最佳 Hit@10 的模型
    - 解釋 Hit@K、MRR、Recall 指標的含義
- 返回字符串形式的完整報告

**main()**
- 調用 generate_analysis_report()
- 將報告保存到 results/analysis_report.txt
- 打印報告到控制台

## 報告格式要求
- 使用繁體中文
- 清晰的部分標題（##）
- 表格格式的結果
- 關鍵發現部分（**關鍵發現**）
- 指標說明和解釋

## 全域約束
- 所有文本和註釋必須使用**繁體中文**
- 代碼變數和函數名保持英文
- 圖表保存為 PNG 格式，DPI 300
- 使用 pandas 處理數據
- 使用 matplotlib 創建視覺化

## 驗收標準
1. visualization.py 實作了所有繪圖函數
2. 可以成功生成三個對比圖表（PNG 文件）
3. 圖表清晰、標題正確、標籤完整
4. analysis.py 成功生成分析報告
5. 報告文件格式正確、內容完整
6. 所有文件都已保存到 results/ 目錄

## 測試命令
```bash
python src/visualization.py
python src/analysis.py
```

應輸出：
- 確認所有 PNG 文件已保存
- 完整的分析報告內容
- 對於分析報告，應顯示每個任務的結果和洞察

## 回報格式
完成後，請在 `.superpowers/task-5-report.md` 中提交報告，包含：
1. 實作的函數概述
2. 生成的圖表列表
3. 報告生成情況
4. 任何問題或疑慮
5. 自審查發現
