# ADE-LLM-S: Supervised Approach for Anomaly Detection and Explanation

This folder contains the implementation of **ADE-LLM-S** (*Anomaly Detection and Explanation using Large Language Models: Supervised Approach*).  
The framework is designed to detect anomalies in **business process traces** and provide **natural language explanations** for their causes.  

At its core, the approach leverages the reasoning capabilities of **large language models (LLMs)**, with a **fine-tuned LLaMA 3.1-8B Instruct model** serving as the main component.  

---

## Framework Overview
![Framework Diagram](supervised.drawio.svg)

The supervised pipeline integrates:
- **Business process simulation**: process models generate realistic normal traces.  
- **Anomaly injection**: controlled deviations mimic common violations, producing labeled anomalous traces.  
- **Data preparation**: combined normal and anomalous traces form a supervised training dataset.  
- **Model fine-tuning**: the LLaMA 3.1-8B Instruct model is adapted using **parameter-efficient fine-tuning techniques**.  
- **Prompt-based inference**: the fine-tuned model classifies traces (normal vs anomalous) and produces **context-aware explanations** that highlight violated process constraints.  

This design ensures the model not only **detects anomalies** but also **explains them in human-understandable terms**, bridging the gap between automated detection and interpretability.  

---

## Folder Structure


