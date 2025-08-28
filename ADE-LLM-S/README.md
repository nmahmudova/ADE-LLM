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
- **conversion/** – Scripts for converting data formats
- **dataset/** – Dataset files used for testing
- **dataset_pre.py** – Preprocessing scripts for datasets
- **eval-BPAD.py** – Script for evaluating the fine-tuned LLaMA model
- **fine-tune-BPAD.py** – Script for fine-tuning LLaMA model on the prepared dataset
- **generation/** – Scripts for generating anomolous traces
- **labelparser/** – Utilities to parse and handle labels in datasets
- **llama-fine-tuned/** – Directory containing fine-tuned LLaMA model checkpoints
- **processmining/** – Scripts related to process simulation
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
- ADE-LLM-S uses the LLaMA 3.1-8B Instruct model as the core component.
- Due to licensing restrictions, the weights are not included in this repository.
- Request access via Hugging Face: [LLaMA 3.1-8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

### 5. Download Process Model Datasets

Download the following datasets and save them under `/dataset/process_model`:

1. **BPM Academic Initiative (BPMAI)**  
   [Download from Zenodo](https://zenodo.org/records/3758705)

2. **Fundamentals of Business Process Management (FBPM)**  
   [Download from FBPM site](http://fundamentals-of-bpm.org/process-model-collections/)

3. **SAP Signavio Academic Models (SAP-SAM)**  
   [Download from Zenodo](https://zenodo.org/records/7012043)

The folder structure should look like this:
- `dataset/process_model/`
  - `BPMAI/`
    - `description.txt`
    - `models/`
  - `FBPM2-ProcessModels/`
    - `Chapter1/`
    - …
    - `Chapter11/`
    - `info.txt`
  - `sap_sam_2022/`
    - `models/`
    - `LICENSE.txt`
    - `README.txt`

### 6. Generate Datasets

To prepare the datasets, run the following command from the root directory:

```bash
python dataset_pre.py
```

We have also provided pre-generated test datasets D1 and D2 in the dataset/ folder:
- test_dataset_1.jsonl corresponds to D1 and is used for evaluating anomaly detection performance.
- test_dataset_cause_1.jsonl contains only the anomalous traces from D1, for assessing the explanation of anomaly causes.
- Similarly, the corresponding files for D2 are also included for testing detection and explanation performance.

### 7. Fine-tune LLaMA (Optional)
You can fine-tune the model if desired. This step can be skipped if you prefer to use our pre-fine-tuned model llama-fine-tuned.
To fine-tune LLaMA from the root directory:
```bash
python fine-tune-BPAD.py
```
This will produce the fine-tuned model llama-fine-tuned ready for inference.

### 8. Evaluate on Test Dataset
To evaluate the model on the provided test datasets, run:
```bash
python eval-BPAD.py
```
This will compute anomaly detection metrics and, if applicable, generate explanations for detected anomalies.

### 9. Running Tests on Your Own Real-World Event Log

You can perform semantic anomaly detection on your own `.xes` event logs using our fine-tuned model (`llama-fine-tuned`). From the root directory, run:

```bash
python test_realLog.py --data_path dataset/BPIC20_PermitLog.xes
```
This command applies the fine-tuned model directly to your specified event log.
We also provide example real-world logs in the dataset/ folder as zip files:
- BPIC20_PermitLog.zip
- Road_Traffic_Fine_Management_Process.zip
After unzipping these files, you will get:
- BPIC20_PermitLog.xes
- Road_Traffic_Fine_Management_Process.xes
These .xes files can be used to test the framework on real-world event data.

### Results

The following tables compare supervised anomaly detection performance between **DABL** and **ADE-LLM-S** on test datasets `D1` (unseen processes) and `D2` (known processes).  
> **Note:** Results are replicated from Wei Guan et al., *"DABL: Detecting Semantic Anomalies in Business Processes Using Large Language Models"* (2024).

#### Test Dataset D1 (Unseen Processes)

| Metric       | DABL | ADE-LLM-S |
|-------------|------|------------|
| Precision (%) | 87.57 | **96.87** |
| Recall (%)    | 91.12 | **92.34** |
| F1-score (%)  | 89.31 | **94.55** |
| Accuracy (%)  | 89.36 | **94.81** |

#### Test Dataset D2 (Known Processes)

| Metric       | DABL | ADE-LLM-S |
|-------------|------|------------|
| Precision (%) | 87.41 | **97.57** |
| Recall (%)    | 87.69 | **93.99** |
| F1-score (%)  | 87.55 | **95.75** |
| Accuracy (%)  | 88.04 | **96.00** |

### Explanation Results

The following tables show **interpretability performance** of ADE-LLM-S compared to DABL in identifying the causes of anomalies, measured using **ROUGE-2** and **ROUGE-L** (Precision, Recall, F1-score).  
> **Note:** DABL results were obtained via replication on the same datasets.

#### Dataset D1

| Method     | ROUGE-2 Precision (%) | ROUGE-2 Recall (%) | ROUGE-2 F1 (%) | ROUGE-L Precision (%) | ROUGE-L Recall (%) | ROUGE-L F1 (%) |
|------------|--------------------|-----------------|---------------|--------------------|-----------------|---------------|
| ADE-LLM-S  | **73.22**           | 70.57           | **71.64**     | **80.09**           | **77.10**       | **78.28**     |
| DABL       | 70.40               | **71.31**       | 70.41         | 75.23               | 76.86           | 75.52         |

#### Dataset D2

| Method     | ROUGE-2 Precision (%) | ROUGE-2 Recall (%) | ROUGE-2 F1 (%) | ROUGE-L Precision (%) | ROUGE-L Recall (%) | ROUGE-L F1 (%) |
|------------|--------------------|-----------------|---------------|--------------------|-----------------|---------------|
| ADE-LLM-S  | **78.33**           | **77.59**       | **77.71**     | **82.79**           | **82.19**       | **82.21**     |
| DABL       | 73.08               | 74.46           | 73.27         | 77.60               | 79.41           | 77.97         |

