# ADE-LLM-S: Supervised Approach for Anomaly Detection and Explanation

This folder contains the implementation of **ADE-LLM-S** (*Anomaly Detection and Explanation using Large Language Models: Supervised Approach*).  
The framework is designed to detect anomalies in **business process traces** and provide **natural language explanations** for their causes.  

At its core, the approach leverages the reasoning capabilities of **large language models (LLMs)**, with a **fine-tuned LLaMA 3.1-8B Instruct model** serving as the main component.  

---

## Framework Overview
<img src="framework.svg" width="800px" alt="Framework Diagram">

The supervised pipeline integrates:
- **Business process simulation**: process models generate realistic normal traces.  
- **Anomaly injection**: controlled deviations mimic common violations, producing labeled anomalous traces.  
- **Data preparation**: combined normal and anomalous traces form a supervised training dataset.  
- **Model fine-tuning**: the LLaMA 3.1-8B Instruct model is adapted using **parameter-efficient fine-tuning techniques**.  
- **Prompt-based inference**: the fine-tuned model classifies traces (normal vs anomalous) and produces **context-aware explanations** that highlight violated process constraints.  

This design ensures the model not only **detects anomalies** but also **explains them in human-understandable terms**, bridging the gap between automated detection and interpretability.  

---

## Folder Structure
- **conversion/** – Scripts or utilities for converting data formats
- **dataset/** – Dataset files used for training and evaluation
- **dataset_pre.py** – Preprocessing scripts for datasets
- **eval-BPAD.py** – Script for evaluating the fine-tuned LLaMA model
- **fine-tune-BPAD.py** – Script for fine-tuning LLaMA model on the prepared dataset
- **generation/** – Scripts for generating traces or examples
- **labelparser/** – Utilities to parse and handle labels in datasets
- **llama-fine-tuned/** – Directory containing fine-tuned LLaMA model checkpoints
- **processmining/** – Scripts related to process simulation or process modeling
- **prompt.py** – Scripts defining prompts for LLaMA inference
- **requirements.txt** – Python dependencies for ADE-LLM-S
- **test_realLog.py** – Script to run inference on real logs
- **utils/** – General utility scripts
- **README.md** – Documentation (this file)

## Setup / Installation

Follow these steps to set up the ADE-LLM-S framework on your machine.

### 1. Clone the Repository
```bash
git clone https://github.com/nmahmudova/ADE-LLM.git
cd ADE-LLM-S
```
### 2. Create a Virtual Environment
It is recommended to use a virtual environment to avoid dependency conflicts.
```bash
python -m venv venv
# Activate the environment:
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```
### 3. Install Dependencies
All required Python packages are listed in requirements.txt.
```bash
pip install -r requirements.txt
```
### 4. Obtain LLaMA Model Weights
ADE-LLM-S uses the LLaMA 3.1-8B Instruct model as the core component.
Due to licensing restrictions, the weights are not included in this repository.
Request access via Meta AI and follow Hugging Face instructions to download and use the model.
