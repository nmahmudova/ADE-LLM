# ADE-LLM-AE: Unsupervised Anomaly Detection in Business Processes

The **ADE-LLM-AE** model addresses the challenge of detecting anomalies in business process data where labeled examples are scarce or unavailable. It is an **unsupervised approach** that does not rely on annotated data, making it suitable for real-world scenarios. While primarily designed as an offline method, the framework incorporates **prefix-based analysis**, allowing early detection of anomalies during process execution. This enables proactive decision-making by flagging irregular behavior before a process trace is fully completed.

## Framework Pipeline

The framework follows a systematic pipeline:

1. **Event Logs** – Raw data capturing business process executions are collected.
2. **Prefix Extraction** – Complete traces are decomposed into partial sequences to enable early anomaly detection.
3. **Encoding** – Symbolic prefixes are transformed into numerical representations suitable for machine learning models.
4. **Autoencoder** – An unsupervised neural network learns to reconstruct normal process behavior from encoded prefixes.
5. **Anomaly Detection** – Deviations are identified by evaluating reconstruction performance, highlighting potential anomalies.
6. **Explanation** – Detected anomalies are interpreted and contextualized in natural language, making results understandable and actionable.

Each step is illustrated with examples to show how raw process data is systematically transformed, modeled, and analyzed, demonstrating not only anomaly detection but also interpretability of results.

## Folder Structure

The project contains the following files:

- **data_loader.py** – Handles loading and preprocessing of raw event log data
- **prepare_prefix_data.py** – Converts complete traces into prefixes  
- **encoder.py** – Encodes prefixes into numerical representations using OneHot or Word2Vec encoding
- **model.py** – Defines the autoencoder model used to learn normal process behavior.  
- **train.py** – Trains the autoencoder on the encoded prefix data.  
- **detect.py** – Uses the trained model to detect anomalies in business process executions.  
- **run_pipeline_results.py** – Runs the full pipeline end-to-end and saves the results into output files.  
- **requirements.txt** – Lists all required Python packages for the project.

## Setup

Follow the steps below to set up the project environment and run the pipeline:

1. **Clone the repository**  
```bash
git clone <repository-url>
cd <repository-folder>
```

2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Run the full pipeline
```bash
python run_pipeline_results.py
```
This will execute all steps, from data preparation and encoding to training, anomaly detection, and explanation generation. Results will be saved in the output files automatically.
