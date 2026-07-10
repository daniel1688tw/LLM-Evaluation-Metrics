#!/bin/bash
# LLM 評估指標 - 完整評估管道進入點
# 依序執行三個評估任務，再產生視覺化與分析報告。
set -e

echo "=========================================="
echo "LLM 評估指標管道"
echo "=========================================="

# 顯示 GPU 狀態（若可用）
python -c "import torch; print('CUDA 可用:', torch.cuda.is_available(), '| 裝置:', (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'))"

echo ""
echo "[1/5] 執行 QA 評估..."
python src/qa_evaluation.py

echo ""
echo "[2/5] 執行文本摘要評估..."
python src/summarization_evaluation.py

echo ""
echo "[3/5] 執行 RAG 評估..."
python src/rag_evaluation.py

echo ""
echo "[4/5] 產生視覺化圖表..."
python src/visualization.py

echo ""
echo "[5/5] 產生分析報告..."
python src/analysis.py

echo ""
echo "=========================================="
echo "✓ 所有評估已完成！"
echo "=========================================="
echo "結果已保存到：results/"
ls -lah results/
