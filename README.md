# ADE-LLM: Anomaly Detection and Explanation using LLMs

Anomaly detection in event logs is crucial for ensuring the reliability, efficiency, and compliance of process-oriented systems such as business workflows, IT operations, and manufacturing pipelines. Semantic anomalies—unexpected or contextually inconsistent activity sequences—are particularly challenging to detect and explain because they rely on process semantics and long-range dependencies. While recent approaches have improved detection accuracy, generating human-understandable explanations remains underexplored.

The **ADE-LLM** framework addresses these challenges by combining anomaly detection with natural language explanations, offering **two complementary methods**:

1. **ADE-LLM-S (Supervised)**  
   - Fine-tunes a LLaMA model on labeled traces.  
   - Focuses on improving both anomaly detection accuracy and the quality of explanations.  

2. **ADE-LLM-AE (Unsupervised Prefix-Based)**  
   - Uses autoencoders trained on trace prefixes to enable early anomaly detection without labeled data.  
   - Leverages zero-shot large language models to generate explanations.  

A central contribution of ADE-LLM is the **systematic comparison of supervised and unsupervised approaches** in terms of detection accuracy, early detection capability, and explanation quality. By integrating advanced language models and emphasizing explainability, ADE-LLM provides more transparent, interpretable, and practically valuable anomaly detection systems in process mining.

## Folder Structure

```text
ADE-LLM/
│
├── ADE-LLM-S/       # Supervised method (see README inside)
├── ADE-LLM-AE/      # Unsupervised prefix-based method (see README inside)
