# LLM 評估指標專案

本專案針對三種自然語言處理任務，對預先訓練的語言模型進行評估，並計算標準評估指標：

| 任務 | 資料集 | 模型（各 2 個） | 指標 |
|------|--------|----------------|------|
| **問答（QA）** | SQuAD | RoBERTa、BERT-large | Exact Match、F1 |
| **文本摘要** | CNN/DailyMail | BART-large、Pegasus | ROUGE-1/2/L、BERTScore |
| **RAG 檢索** | HippoCamp | MiniLM、BGE-small（retriever） | Hit@1/5/10、MRR、file-level Recall |

所有評估皆在 **Docker 容器 + 本機 NVIDIA GPU** 上執行真實模型推理（非模擬資料）。

---

## 執行環境需求

- [Docker](https://www.docker.com/)（含 Docker Desktop，Windows 建議搭配 WSL2 後端）
- **NVIDIA GPU** 與 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)（讓容器能存取 GPU）
  - 本專案開發環境為 NVIDIA RTX 4050 Laptop（6GB VRAM），CUDA 13.0 驅動
- 首次執行需能連上網際網路，以下載模型與資料集

> **為何一定要用 GPU？** 純 CPU 上跑真實推理極慢（QA 任務在 CPU 上曾耗時約 5 小時）。改用 GPU 後同樣任務數分鐘即可完成。

---

## 快速開始

### 1. 建置 Docker 映像

```bash
docker build -t llm-eval:gpu .
```

映像以 `pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime` 為基礎（內建 torch 2.1.0 + CUDA 11.8，支援 Ada 架構 GPU），並安裝評估所需套件與繁體中文字型。

### 2. 執行完整評估管道

```bash
docker run --rm --gpus all \
  -v "$(pwd)/results:/workspace/results" \
  -v "$(pwd)/.hf_cache:/workspace/.hf_cache" \
  llm-eval:gpu
```

預設會執行 `entrypoint.sh`，依序完成：QA 評估 → 摘要評估 → RAG 評估 → 產生圖表 → 產生分析報告。
- `-v .../results`：將結果輸出到本機 `results/` 目錄
- `-v .../.hf_cache`：快取下載的模型與資料集，之後重跑不需重新下載

> Windows PowerShell 請將 `$(pwd)` 改為 `${PWD}`。

### 3. 單獨執行某一項任務（可選）

```bash
docker run --rm --gpus all -v "$(pwd)/results:/workspace/results" -v "$(pwd)/.hf_cache:/workspace/.hf_cache" llm-eval:gpu python src/qa_evaluation.py
docker run --rm --gpus all -v "$(pwd)/results:/workspace/results" -v "$(pwd)/.hf_cache:/workspace/.hf_cache" llm-eval:gpu python src/summarization_evaluation.py
docker run --rm --gpus all -v "$(pwd)/results:/workspace/results" -v "$(pwd)/.hf_cache:/workspace/.hf_cache" llm-eval:gpu python src/rag_evaluation.py
```

---

## 檔案結構

```
.
├── Dockerfile              # GPU 版映像（PyTorch CUDA 基礎映像 + Noto CJK 字型）
├── entrypoint.sh           # 完整評估管道（依序執行 5 個步驟）
├── requirements.txt        # Python 依賴（torch 由基礎映像提供，故清單中不含）
├── config.json             # 三個任務的模型與資料集設定
├── .dockerignore
├── src/
│   ├── metrics_utils.py            # QA（normalize/EM/F1）與摘要（ROUGE/BERTScore）指標
│   ├── qa_evaluation.py            # QA 評估主程式
│   ├── summarization_evaluation.py # 摘要評估主程式
│   ├── rag_utils.py                # RAG 指標（Hit@K/MRR/file-level Recall）
│   ├── rag_evaluation.py           # RAG 檢索評估主程式
│   ├── visualization.py            # 產生模型對比長條圖
│   └── analysis.py                 # 產生彙總分析報告
├── results/                # 輸出目錄（執行後產生，已被 .gitignore 排除）
├── docs/                   # 實作計畫等文件
├── RESULTS.md              # 評估結果彙總
└── ANALYSIS.md             # 指標說明、模型對比與實驗設計反思
```

---

## 輸出說明

執行後 `results/` 目錄會產生：

| 檔案 | 說明 |
|------|------|
| `qa_results.csv` / `.json` | QA 評估結果 |
| `summarization_results.csv` / `.json` | 摘要評估結果 |
| `rag_results.csv` / `.json` | RAG 檢索評估結果 |
| `qa_comparison.png` | QA 模型對比圖 |
| `summarization_comparison.png` | 摘要模型對比圖 |
| `rag_comparison.png` | RAG 模型對比圖 |
| `analysis_report.txt` | 彙總分析報告（繁體中文） |

實際評估數字請見 [RESULTS.md](RESULTS.md)。

---

## 評估指標定義

### QA
- **Exact Match（EM）**：模型答案與標準答案（正規化後）逐字完全相同才計分，屬嚴格度量。
- **F1 Score**：預測答案與標準答案在詞元（token）層級的重疊，計算精確率與召回率的調和平均，屬較寬鬆度量。

### 文本摘要
- **ROUGE-1 / ROUGE-2 / ROUGE-L**：生成摘要與參考摘要在 unigram / bigram / 最長公共子序列上的重疊。
- **BERTScore**：使用 BERT 上下文詞嵌入計算語義相似度，能捕捉「用詞不同但語意相近」的情況。

### RAG 檢索
- **Hit@K**：前 K 個檢索結果中，是否至少命中一個正確證據檔案。
- **MRR（Mean Reciprocal Rank）**：每個查詢第一個命中結果排名的倒數，取平均。
- **file-level Recall@10**：前 10 個結果涵蓋了多少比例的正確證據檔案。

---

## 參考資源

- SQuAD：<https://rajpurkar.github.io/SQuAD-explorer/>
- CNN/DailyMail：<https://huggingface.co/datasets/cnn_dailymail>
- HippoCamp：<https://hippocamp-ai.github.io/>、<https://huggingface.co/datasets/MMMem-org/HippoCamp>
- ROUGE：<https://aclanthology.org/W04-1013/>
- BERTScore：<https://arxiv.org/abs/1904.09675>
- Sentence-Transformers：<https://www.sbert.net/>
- BGE 嵌入模型：<https://huggingface.co/BAAI/bge-small-en-v1.5>
