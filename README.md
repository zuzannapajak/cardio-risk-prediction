# 🫀 Cardiovascular Disease Risk Prediction using Machine Learning

This project focuses on **predicting the risk of cardiovascular disease (CVD)** using a variety of supervised learning algorithms, including **Random Forest**, **AdaBoost**, **XGBoost**, **Naive Bayes**, **Support Vector Machine (SVM)**, and **Logistic Regression**.  
It also explores **ensemble methods** such as **Stacking** and **Soft Voting**, along with dimensionality reduction techniques like **PCA**, **t-SNE**, and **UMAP**.  

The workflow covers the entire pipeline — from **data cleaning**, **exploratory data analysis (EDA)**, and **preprocessing**, to **model training**, **evaluation**, **clustering**, and **interpretation** of results.

## 🎯 Objective & Motivation

Cardiovascular diseases constitute the **leading cause of death worldwide**.  
Early detection of patients at high risk can significantly improve outcomes of the treatment and visibly reduce mortality rates.  
The goal of this project is to develop a **data-driven machine learning pipeline** that can analyze key clinical indicators and **predict the likelihood of heart failure**. Through interpretable and reproducible models medical decision-making will be supported even stronger. 

## 📁 Project Structure

```
CARDIO-RISK-PREDICTION/
│
├── data/                   # Contains the dataset (heart.csv)
│ └── heart.csv             # Original Heart Failure Clinical Records dataset
│
├── notebooks/              # Jupyter notebooks for analysis and experimentation
│ ├── eda.ipynb             # Exploratory Data Analysis (EDA)
│ ├── dim_red.ipynb         # Dimensionality reduction (PCA, t-SNE, UMAP)
│ ├── clustering.ipynb      # Unsupervised clustering and visualization
│ └── model_training.ipynb  # Model training, evaluation, and comparison
│
├── outputs/                # Generated results, and trained models
│ └── (files generated during analysis)
│
├── scripts/                # Modular Python source code for reproducibility
│ │
│ ├── eda/                  # Scripts supporting data exploration
│ │ ├── cleaning/           # Data cleaning and preprocessing utilities
│ │ │ └── init.py
│ │ ├── preprocessing/      # Custom preprocessing pipelines and transformers
│ │ │ ├── pipeline.py       # End-to-end preprocessing pipeline
│ │ │ ├── transformers.py   # Custom data transformation classes
│ │ ├── plots.py            # Visualization utilities for EDA
│ │ ├── reporting.py        # Summary tables and dataset reports
│ │ └── stats.py            # Statistical analysis functions
│ │
│ ├── dim_red/              # Dimensionality reduction utilities
│ │ └── plots.py            # Visualization of reduced feature spaces
│ │
│ ├── clustering/           # Unsupervised clustering methods and tools
│ │ └── utils.py            # Clustering metrics and helper functions
│ │
│ └── model_training/       # Model training and evaluation scripts
│ └── utils.py              # Functions for evaluation, and metrics
│
├── venv/                   # Virtual environment for project dependencies
│
├── .gitignore              # Git ignore file for virtual environment and cache
├── LICENSE                 # Project license file
├── requirements.txt        # List of Python dependencies
└── README.md               # Project documentation
```


## 📊 Dataset

The project uses the **Heart Failure Clinical Records Dataset**, originally published on **Kaggle**:

🔗 [Kaggle – Heart Failure Clinical Data](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data)

### Dataset Overview
- **Rows:** 299 patients  
- **Features:** 13 clinical variables + target  
- **Target variable:** `DEATH_EVENT` (1 = death occurred, 0 = survived)  
- **Attributes include:**
  - `age` – Age of the patient  
  - `anaemia` – Decrease of red blood cells or hemoglobin (boolean)  
  - `creatinine_phosphokinase` – CPK enzyme level (mcg/L)  
  - `diabetes` – Whether the patient has diabetes (boolean)  
  - `ejection_fraction` – Percentage of blood leaving the heart per contraction  
  - `high_blood_pressure` – Whether the patient has hypertension (boolean)  
  - `platelets` – Platelet count (kiloplatelets/mL)  
  - `serum_creatinine` – Level of serum creatinine (mg/dL)  
  - `serum_sodium` – Level of serum sodium (mEq/L)  
  - `sex` – Male or Female (binary)  
  - `smoking` – Whether the patient smokes (boolean)  
  - `time` – Follow-up period (days)  

### License
Data files © Original authors (as stated on the Kaggle dataset page).  
Used here for **academic and research purposes only**.

## 🛠 Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/zuzannapajak/cardio-risk-prediction.git
cd cardio-risk-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 📘 Usage

`notebooks/eda.ipynb` – Perform exploratory data analysis  
`notebooks/dim_red.ipynb` – Apply PCA, t-SNE, and UMAP  
`notebooks/model_training.ipynb` – Train and evaluate classification models  
`notebooks/clustering.ipynb` – Visualize unsupervised grouping patterns  

Outputs (e.g. trained models) are automatically saved in the `outputs/` directory.


## 🧠 Models Used

**Classification Models**
- Random Forest
- AdaBoost
- XGBoost
- Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression

**Ensemble Methods**
- Stacking Classifier
- Soft Voting Classifier

**Dimensionality Reduction**
- Principal Component Analysis (PCA)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Uniform Manifold Approximation and Projection (UMAP)

**Clustering**
- K-Means
- Agglomerative Clustering
- DBSCAN


## 🔄 Workflow Overview
1. **Data Cleaning & EDA** – Analyze data distribution, missing values, and correlations
2. **Preprocessing Pipeline** – Apply normalization, encoding, and transformations
3. **Dimensionality Reduction** – Visualize structure in low-dimensional space
4. **Model Training & Evaluation** – Train and compare multiple ML models
5. **Clustering & Insights** – Explore patterns and patient group similarities

## 📈 Evaluation Metrics
- Accuracy
- Balanced accuracy
- Precision (pos=1)
- Recall / Sensitivity (TPR)
- Specificity (TNR)
- F1
- F0.5
- F2
- ROC-AUC
- PR-AUC (Average Precision)
- Log loss
- Jaccard
- Hamming loss
- Matthews Correlation Coefficient (MCC)
- Cohen’s kappa
- Zero-One loss
- Brier score
- Threshold
- Positives (TP+FN), Negatives (TN+FP)
- Confusion matrix

## 🏆 Results Summary

| Model              | Threshold | F1 Score | MCC   | Balanced Accuracy | ROC-AUC | PR-AUC | Accuracy | False Positive | False Negative |
|--------------------|------------|----------|-------|-------------------|---------|--------|-----------|----------------|----------------|
| **Voting Ensemble**   | 0.304     | **0.913** | **0.871** | **0.951** | **0.981** | **0.963** | **0.936** | 4 | 0 |
| **Stacking Ensemble** | 0.540     | 0.910 | 0.861 | 0.940 | 0.980 | 0.959 | 0.935 | 3 | 1 |
| Logistic Regression   | 0.633     | 0.905 | 0.856 | 0.928 | 0.968 | 0.931 | 0.935 | 2 | 2 |
| SVM                   | 0.452     | 0.857 | 0.784 | 0.892 | 0.949 | 0.920 | 0.903 | 3 | 3 |
| XGBoost               | 0.522     | 0.850 | 0.781 | 0.880 | 0.965 | 0.913 | 0.903 | 2 | 4 |
| Naive Bayes           | 0.380     | 0.818 | 0.720 | 0.868 | 0.931 | 0.844 | 0.871 | 5 | 3 |
| Random Forest         | 0.561     | 0.811 | 0.746 | 0.845 | 0.985 | **0.970** | 0.887 | 1 | 6 |
| AdaBoost              | 0.881     | 0.783 | 0.662 | 0.843 | 0.843 | 0.666 | 0.838 | 7 | 3 |

✅ **Best Overall Model:** *Voting Ensemble* — achieved the highest F1 score (0.913), ROC-AUC (0.981), and balanced accuracy (0.951), with **zero false negatives**.  
This indicates **excellent sensitivity** and very strong generalization on this dataset.

## 🚀 Future Work
- Integrate SHAP and LIME for explainable AI insights
- Extend dataset with additional clinical variables or multi-center data
- Explore deep learning architectures (e.g., MLP, TabNet)
- Develop a web-based dashboard for real-time CVD risk prediction
- Implement cross-validation visualization and model interpretability modules


## 📚 Citation

If you reference this project in your research, please cite it as:

> Pajak, Z. (2025). *Machine Learning-Based Risk Prediction of Cardiovascular Diseases Using Ensemble Models and Dimensionality Reduction Techniques*.  
> Bachelor’s Thesis, Technical University of Łódź (TUL), Poland.  
> GitHub: [https://github.com/zuzannapajak/cardio-risk-prediction](https://github.com/zuzannapajak/cardio-risk-prediction)

---

*Author: Zuzanna Pajak*  
*Bachelor’s Thesis – Technical University of Łódź (TUL), Faculty of Electrical, Electronic, Computer and Control Engineering*  
*© 2025 – Academic project for research and educational purposes.*
