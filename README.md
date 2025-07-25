# Cardiovascular Disease Risk Prediction using Machine Learning

This project aims to predict the risk of cardiovascular diseases (CVD) using machine learning techniques such as Random Forest, AdaBoost, XGBoost, and dimensionality reduction (PCA, t-SNE, UMAP). It includes thorough data preprocessing, exploratory data analysis (EDA), model training, evaluation, clustering, and visualization.

## 📁 Project Structure

```
cardio-risk-prediction/
├── data/                   # Original and cleaned datasets
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── scripts/                # Python scripts for modular tasks (e.g. preprocessing, training)
├── outputs/                # Generated outputs (figures, models, etc.)
├── requirements.txt        # List of required Python packages
├── README.md               # Project overview and instructions
└── venv/                   # Virtual environment (optional)
```

## 📊 Dataset

The dataset used is related to cardiovascular health, including clinical and lifestyle-related features.

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

- Use `notebooks/eda.ipynb` to perform EDA and visualization.
- Run `scripts/preprocessing.py` to clean and prepare your dataset.
- Run `scripts/training.py` to train and evaluate your models.

## 🧠 Models Used

- Random Forest
- AdaBoost
- XGBoost
- Dimensionality Reduction: PCA, t-SNE, UMAP
- Clustering: KMeans, Hierarchical Clustering

## 📈 Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC Curve and AUC
- Confusion Matrix

---

*Author: Zuzanna Pajak*