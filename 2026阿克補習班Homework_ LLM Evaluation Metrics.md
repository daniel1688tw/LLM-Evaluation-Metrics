# 2026阿克補習班Homework: LLM Evaluation Metrics
Speaker: 陳旻寬
Slide link: https://docs.google.com/presentation/d/1PLk7mr1L_K_PoM8nZtn0XoNtCL2JK5kK/edit?usp=drive_link&ouid=111067217030326601684&rtpof=true&sd=true

This assignment aims to provide hands-on experience in evaluating Large Language Models (LLMs) for **Question Answering (QA)**, **Text Summarization** and **RAG**. You will learn to apply standard evaluation metrics and analyze model performance using pre-defined benchmarks and models.

:::warning
:warning: You can use AI agents (such as Claude Code) to complete tasks, but remember to always remain skeptical and verify information. Do not blindly trust or agree to every action and report conclusion of the AI agent.
:::

## Requirements

1.  **Question Answering (QA) Model Evaluation:**
    * **Task:** Evaluate pre-trained LLMs on a QA benchmark dataset.
    * **Benchmark Dataset:** Download from huggingface or other opensource.
    * **Model Selection:** Choose at least **two (2)** pre-trained QA models from the Hugging Face `transformers` library (e.g. `deepset/roberta-base-squad2`).
    * **Metrics:** Calculate and report **Exact Match (Accuracy)** and **F1 Score** for each model.
    * **Analysis:** Briefly discuss the meaning of these metrics and compare the performance of your chosen models.

2.  **Text Summarization Model Evaluation:**
    * **Task:** Evaluate pre-trained LLMs on a text summarization benchmark dataset.
    * **Benchmark Dataset:** Download from huggingface or other opensource.
    * **Model Selection:** Choose at least **two (2)** pre-trained summarization models from the Hugging Face `transformers` library (e.g. `facebook/bart-large-cnn`).
    * **Metrics:** Calculate and report **ROUGE scores** (ROUGE-1, ROUGE-2, ROUGE-L) and **BERTScore** for each model.
    * **Analysis:** Briefly discuss the meaning of these metrics and compare the performance of your chosen models.

3. **RAG Evaluation:**
    * **Task:** Evaluate pre-trained LLMs on RAG.
    * **Benchmark Dataset:** HippoCamp: Benchmarking Contextual Agents on Personal Computers (https://hippocamp-ai.github.io/)
    * **Model Selection:** Choose at least **two** pre-trained summarization models from the Hugging Face `transformers` library (e.g. `facebook/bart-large-cnn`).
    * **Metrics:** Calculate and report **Hit@1/5/10**, **MRR**, and **file-level Recall** for each model.
    * **Analysis:** Briefly discuss the meaning of these metrics and compare the performance of your chosen models.

## Checklist

* **Submission Method:**
    * [ ] Submit your work via GitLab/ GitHub/ CodiMD/ HackMD.

* **Environment & Setup:**
    * [ ] Your code must be runnable within a Docker container on our lab server or somewhere.

* **For the QA Task:**
    * [ ] Download and briefly describe the chosen QA dataset.
    * [ ] Successfully run inference with your selected QA LLMs.
    * [ ] Present the results for Exact Match (Accuracy) and F1 Score in a table or plot.

* **For the Summarization Task:**
    * [ ] Download and briefly describe the chosen summarization dataset.
    * [ ] Successfully run inference with your selected summarization LLMs.
    * [ ] Present the ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) and BERTScore results in a table or plot.

* **For the RAG Task:**
    * [ ] Download and briefly describe the chosen summarization dataset.
    * [ ] Successfully run inference with your selected summarization LLMs.
    * [ ] Present the **Hit@1/5/10**, **MRR**, and **file-level Recall** results in a table or plot.

* **Optional:**
    * [ ] Share any key insights or reflections gained from this assignment, such as observations on experimental design, benchmarks, or feedback regarding LLM capabilities. (依照經驗，老師會更想要知道你從中得到甚麼啟發)

