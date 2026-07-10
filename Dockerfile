# LLM 評估指標專案 Docker 配置

FROM python:3.9-slim

# 設定工作目錄
WORKDIR /workspace

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=False

# 複製依賴清單並安裝
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案代碼
COPY . .

# 預設命令：支持執行評估腳本
# 用戶可以通過以下方式使用：
# docker run <image> python scripts/evaluate.py
# docker run <image> python scripts/run_evaluation.py
CMD ["python", "-m", "pip", "list"]
