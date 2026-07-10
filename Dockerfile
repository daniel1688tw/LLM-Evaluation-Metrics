# LLM 評估指標專案 Docker 配置（GPU 版）
#
# 使用 PyTorch 官方 CUDA 映像作為基礎，內建 torch 2.1.0 + CUDA 11.8，
# CUDA 11.8 原生支援 Ada 架構（RTX 40 系列，如 RTX 4050，compute 8.9），
# 可搭配 `docker run --gpus all` 使用本機 NVIDIA GPU 加速推理。

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 設定工作目錄
WORKDIR /workspace

# 設定環境變數
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=False
# 將 Hugging Face 模型/資料集快取放到可掛載的目錄，避免每次重跑都重新下載
ENV HF_HOME=/workspace/.hf_cache

# 複製依賴清單並安裝（torch 已由基礎映像提供，requirements.txt 不含 torch）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案代碼
COPY . .

# 建立結果目錄
RUN mkdir -p results

# 預設命令：執行完整評估管道
CMD ["bash", "entrypoint.sh"]
