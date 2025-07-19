**LLM‑Powered Fault Diagnosis & Fix Recommender**  

**Prototype Documentation**

By: Byaivab Sarkar  

Date: 19 July 2025

This document describes the end‑to‑end development of a prototype system—**“LLM‑Powered Fault Diagnosis & Fix Recommender”**—designed to demonstrate a closed‑loop decision‑making capability for facility management. Combining anomaly detection with large‑language‑model reasoning, the prototype flags sensor anomalies, diagnoses probable causes, recommends corrective actions, and generates technician‑ready reports, all within seconds. This aligns directly with Xempla’s vision of an **Autonomous Maintenance Operating Center** that “Thinks, Plans, and Acts—So You Don’t Have To.”

## <a name="_4hygy1dyb27"></a>**1.Objectives & Alignment with Xempla**
1. **Autonomous Fault Management**
   Showcase how ML and LLMs can be chained to detect, diagnose, and remediate equipment issues without manual intervention.
1. **Operational Efficiency**
   Reduce mean time to detect (MTTD) and mean time to repair (MTTR) by automating root‑cause reasoning and guidance.
1. **Explainable AI**
   Provide clear, human‑readable explanations and step‑by‑step recommendations for maintenance personnel.
1. **Scalability & Closed‑Loop Learning**
   Lay the groundwork for feedback integration—technician actions and outcomes feed back into model improvements.
## <a name="_8bdr2bkw3rrl"></a>**2. System Architecture**
1. **Data Ingestion**
   Raw sensor streams (temperature, pressure, vibration) at 1‑minute intervals
   Synthetic dataset (100,000 samples) with injected anomalies for testing
1. **Anomaly Detection Engine**
   Isolation Forest (scikit‑learn) trained on historical normal data
   Flags outliers with anomaly scores
1. **LLM Decision Agent**
   LangChain orchestration
   Single‑call prompt combining diagnosis, recommendation, and summary stages
   OpenAI GPT‑3.5‑turbo (or configurable model)
1. **Reporting & UI**
   Streamlit web app
   Interactive chart of sensor streams with anomalies highlighted
   “Diagnose & Recommend” button triggers LLM call
   Displays cause, fix steps, and formatted technician report
1. **(Future) Feedback Loop**
   Capture technician acceptance/rejection of recommendations
   Store in a database for periodic retraining of anomaly thresholds and prompt refinement
-----
## <a name="_ilblvoy87r3u"></a>**3. Data Preparation & Synthetic Dataset**
- **Dataset**: 100,000 rows over ~70 days at 1‑minute frequency
- **Features**:
  timestamp (datetime)
  temperature(C) (normal ≈ 28 ± 1.5°C; spikes of +10–15°C every 2,000 rows)
  pressure(bar) (normal ≈ 1.2 ± 0.1 bar; drops of 0.5–0.7 bar at anomalies)
  vibration(mm/s) (normal ≈ 2.0 ± 0.3 mm/s; spikes of +3–5 mm/s at anomalies)
- **Usage**: Serves as stand‑in for real facility data, enabling rapid prototyping without access constraints.
## <a name="_f8apuud8r2rx"></a>**4. Anomaly Detection Module**
1. **Algorithm Choice**
   Isolation Forest for its unsupervised, fast, and interpretable nature on tabular data.
1. **Implementation Highlights**
   Trained on a sliding window of historical “normal” data
   Outputs an anomaly score; values above 0.85 flagged
1. **Evaluation & Visualization**
   Jupyter notebook (notebooks/anomaly\_detection.ipynb) documents EDA, training, and anomaly plots.

**5. LLM‑Powered Decision Agent**

1. **Prompt Engineering
   Single consolidated prompt** containing:
   Anomaly values
   Instructions to diagnose, recommend, and summarize
1. **LangChain Setup**
   One LLMChain object using gpt-3.5-turbo
   Parameters: temperature=0.3, max\_tokens=300, streaming=True
## <a name="_vpc66djv3fze"></a>**6. Streamlit Application**
1. **File**: app/app.py
1. **Key Features**
   Sidebar: API key configuration, model selection (GPT‑3.5 vs. local)
   Main panel:
   `	`Data uploader (or default synthetic file)
   `	`Line chart with anomaly highlights
   `	`“Run Diagnosis” button
   `	`Text area for LLM report
1. **User Flow**
   `	`Load data → view plot
   `	`Click Diagnose → spinner indicates LLM call
   `	`Report appears within 30-50 seconds (in optimized mode)
## <a name="_wydb2qj4p13k"></a>**7. Integration with Xempla Toolkit**
1. **Code Modules** (llm/, app/, notebooks/) can be refactored into Xempla’s internal **Decision Intelligence Toolkit**.
1. **Prompt templates** and **caching patterns** become reusable components across other asset types.
1. **API wrappers** can be containerized (Docker) and deployed at edge hubs for low‑latency inference.
## <a name="_3kttzixotlw9"></a>**8. References**
- Xempla Autonomous Maintenance Operating Center overview:[ ](https://xempla.io)[*https://xempla.io*
  ](https://xempla.io)
- Scikit‑learn Isolation Forest doc
