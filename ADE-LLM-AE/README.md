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

