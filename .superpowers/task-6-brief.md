# 任務 6：Docker 設置與測試

## 目標
完成 Docker 配置並測試完整的評估管道。更新 Dockerfile、建立 entrypoint.sh 腳本，並驗證在 Docker 容器中可以成功運行完整的評估流程。

## 需要建立/修改的文件

### 1. Dockerfile（修改）
更新現有的 Dockerfile，確保可以執行完整的評估管道。

要求：
- 基礎映像：python:3.9-slim
- 工作目錄：/workspace
- 安裝依賴：pip install -r requirements.txt
- 複製項目代碼
- 建立 results 目錄
- 設定環境變數：
  - PYTHONUNBUFFERED=1
  - HF_HUB_OFFLINE=False
- 設定默認命令為執行 entrypoint.sh

### 2. entrypoint.sh（建立）
建立 bash 腳本來順序執行所有評估任務。

腳本應該：
- 使用 `set -e` 確保任何錯誤都會停止執行
- 打印清晰的進度消息（繁體中文）
- 順序執行以下命令：
  1. `python src/qa_evaluation.py`
  2. `python src/summarization_evaluation.py`
  3. `python src/rag_evaluation.py`
  4. `python src/visualization.py`
  5. `python src/analysis.py`
- 在每個步驟之前打印進度號（[1/5]、[2/5] 等）
- 在完成時打印成功消息
- 最後列出 results/ 目錄中的所有文件

腳本示例結構：
```bash
#!/bin/bash
set -e

echo "=========================================="
echo "LLM 評估指標管道"
echo "=========================================="

echo ""
echo "[1/5] 執行 QA 評估..."
python src/qa_evaluation.py

echo ""
echo "[2/5] 執行摘要評估..."
python src/summarization_evaluation.py

... (更多步驟)

echo ""
echo "=========================================="
echo "✓ 所有評估已完成！"
echo "=========================================="
echo "結果已保存到：results/"
ls -lah results/
```

### Docker 環境驗證
在 Dockerfile 中驗證：
- PyTorch 是否可以檢測到 CUDA（如果可用）
- 所有依賴都能正確導入

## 測試要求

### 本地構建測試
1. 成功構建 Docker 映像
2. 映像大小應該合理
3. 沒有構建警告或錯誤

### 容器執行測試
1. 啟動容器
2. 確保所有五個評估步驟都按順序執行
3. 驗證生成了所有預期的結果文件：
   - qa_results.csv、qa_results.json、qa_comparison.png
   - summarization_results.csv、summarization_results.json、summarization_comparison.png
   - rag_results.csv、rag_results.json、rag_comparison.png
   - analysis_report.txt

### 文件卷掛載測試
確保可以：
1. 將本地 results/ 目錄掛載到容器
2. 成功讀取和寫入結果文件
3. 容器完成後，本地文件系統中存在結果

## 全域約束
- 所有文檔和指令必須使用**繁體中文**
- Dockerfile 應該遵循最佳做法
- entrypoint.sh 應該是可執行的
- 不要在容器內進行不必要的操作

## 驗收標準
1. Dockerfile 已更新，包含所有必要的配置
2. entrypoint.sh 已建立並可執行（chmod +x）
3. Docker 映像可以成功構建
4. 容器可以按順序執行所有五個步驟
5. 所有結果文件都已生成
6. 可以從本地文件系統訪問 results/ 目錄中的文件

## 測試命令
```bash
# 構建映像
docker build -t llm-evaluation-metrics:latest .

# 運行容器（本地）
docker run -v $(pwd)/results:/workspace/results llm-evaluation-metrics:latest

# 或在 Windows PowerShell 中
docker run -v ${PWD}\results:/workspace/results llm-evaluation-metrics:latest

# 驗證結果
ls -la results/
```

## 回報格式
完成後，請在 `.superpowers/task-6-report.md` 中提交報告，包含：
1. Dockerfile 和 entrypoint.sh 的修改摘要
2. Docker 構建結果（成功/失敗）
3. 容器執行結果（所有步驟的輸出摘要）
4. 生成的文件清單
5. 任何問題或疑慮
6. 自審查發現
