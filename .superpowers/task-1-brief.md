# 任務 1：環境設置與依賴管理

## 目標
建立專案的基礎環境和配置文件：建立 `requirements.txt`、`Dockerfile`、`config.json` 和 `.gitignore`。

## 需要建立的文件

### 1. requirements.txt
Python 套件依賴清單。包含以下主要套件：
- torch==2.0.1
- transformers==4.30.2
- datasets==2.13.0
- evaluate==0.4.0
- scikit-learn==1.3.0
- pandas==2.0.3
- matplotlib==3.7.2
- numpy==1.24.3
- tqdm==4.65.0
- requests==2.31.0

### 2. Dockerfile
用於容器化專案的 Docker 配置。基礎映像：python:3.9-slim
- 工作目錄：/workspace
- 複製 requirements.txt 並安裝依賴
- 複製專案代碼
- 設定環境變數：PYTHONUNBUFFERED=1、HF_HUB_OFFLINE=False
- 預設命令應支持執行評估腳本

### 3. config.json
模型和數據集配置文件。結構如下：
```json
{
  "qa": {
    "dataset": {
      "name": "squad",
      "split": "validation"
    },
    "models": [
      "deepset/roberta-base-squad2",
      "bert-large-uncased-whole-word-masking-finetuned-squad"
    ]
  },
  "summarization": {
    "dataset": {
      "name": "cnn_dailymail",
      "config": "3.0.0",
      "split": "validation"
    },
    "models": [
      "facebook/bart-large-cnn",
      "google/pegasus-cnn_dailymail"
    ]
  },
  "rag": {
    "dataset": {
      "name": "hippocamp",
      "url": "https://hippocamp-ai.github.io/"
    },
    "models": [
      "facebook/bart-large-cnn",
      "google/pegasus-cnn_dailymail"
    ]
  }
}
```

### 4. .gitignore
排除 Python 緩存、結果目錄和臨時文件：
```
__pycache__/
*.pyc
.DS_Store
results/
*.log
.ipynb_checkpoints/
cache/
*.egg-info/
dist/
build/
.venv/
venv/
*.egg
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.zip
*.tar.gz
```

## 全域約束
- 所有文件必須使用繁體中文編寫（註釋、文檔等）
- 代碼本身保持英文標準
- 建立的所有文件應按照指定的位置和格式

## 驗收標準
1. 所有四個文件都已建立
2. requirements.txt 包含完整的依賴清單
3. Dockerfile 可以成功建置，無錯誤
4. config.json 格式正確，可被 Python json 模塊解析
5. .gitignore 格式正確

## 回報格式
請在完成時提交一個報告，包含：
1. 各文件的建立狀態
2. 每個文件的驗證結果
3. 任何構建或驗證過程中遇到的問題
