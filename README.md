# Conversation Summary Clustering and Error Taxonomy Project

## 📌 Project Purpose

This project processes and analyzes conversation summaries collected from the **Agent Assist tool**.  
It is designed to handle **15,000+ records**, cluster conversations with machine learning algorithms, and apply a structured error taxonomy to identify common failure types and conversation patterns.

The overall goal is to **clean, cluster, and label support interactions** in order to generate insights into recurring issues, agent performance, and opportunities for operational improvement.

---

## ⚙️ What the Project Does

- **Data Cleaning & Preparation**  
  Cleans raw conversation summaries and enriches them with features such as length and time.

- **Clustering & Topic Modeling**  
  Groups conversations into clusters and topics using machine learning algorithms.

- **Error Taxonomy Labeling**  
  Applies a curated taxonomy to classify conversation errors into categories (e.g., agent, process, system, communication).

- **Semantic Analysis**  
  Uses transformer-based models and LLMs to identify and label error categories from summaries.

- **Batch & Parallel Processing**  
  Supports large-scale error analysis across thousands of records efficiently.

- **Interactive Visualization**  
  A Streamlit dashboard for exploring conversation trends, error rates, agent workload, and key performance metrics.

---

## 📂 Main Files and Their Purpose

- `topic_modeling_base.ipynb` – Base topic modeling and clustering experiments  
- `topic_modeling_error_taxonomy.ipynb` – Error taxonomy extraction and application  
- `clustering_algorithm.ipynb` – Testing and comparing clustering approaches  
- `error_labeler.ipynb` – Labeling conversation errors using semantic similarity  
- `data_pipeline.py` – Data cleaning and feature engineering pipeline  
- `analysis_error_finding.py` – Batch and parallel error analysis with taxonomy extraction  
- `label_model.py` – Transformer-based taxonomy labeler implementation  
- `dashboard.py` – Streamlit app for visualization of conversation analytics  

---

## 🔄 Typical Workflow

1. **Prepare and clean data** with `data_pipeline.py`
2. **Explore clustering and topic modeling** in the Jupyter notebooks
3. **Extract and label errors** with `analysis_error_finding.py` and `label_model.py`
4. **Visualize results** in the interactive Streamlit dashboard (`dashboard.py`)

---

## 🎯 Intended Use

- Analyze and clean conversation data from the Agent Assist tool  
- Identify frequent error types and recurring conversation topics  
- Provide **interactive dashboards** for stakeholders to explore support insights  
- Support **continuous improvement** in agent performance and customer support quality  

---
