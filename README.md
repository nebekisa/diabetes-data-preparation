
# Diabetes Prediction Data Preparation Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

A comprehensive medical data preparation pipeline for diabetes prediction, transforming raw healthcare data into machine-learning ready datasets with clinical validation and optimal feature engineering.

## 🎯 Project Overview

Diabetes is a growing global health concern affecting millions worldwide. This project addresses the critical need for clean, well-structured medical data to support accurate predictive analytics in healthcare. The pipeline transforms challenging medical datasets with data quality issues into robust, clinically relevant resources for diabetes prediction.

### Key Achievements
- **31% improvement** in diabetic case detection (53.7% → 70.4% sensitivity)
- **48.7% missing data** handled in insulin measurements
- **15 optimal features** selected from original 32 through consensus methods
- **Clinical validation** of all engineered features and transformations

## 📊 Dataset

**Source**: UCI Machine Learning Repository - Pima Indians Diabetes Database

**Original Features**:
- `Pregnancies`: Number of times pregnant
- `Glucose`: Plasma glucose concentration (mg/dL)
- `BloodPressure`: Diastolic blood pressure (mm Hg)
- `SkinThickness`: Triceps skin fold thickness (mm)
- `Insulin`: 2-Hour serum insulin (mu U/ml)
- `BMI`: Body mass index (kg/m²)
- `DiabetesPedigreeFunction`: Diabetes pedigree function
- `Age`: Age in years
- `Outcome`: Target variable (0 = non-diabetic, 1 = diabetic)

## 🏗️ Project Architecture
diabetes-data-preparation/
│
├── data/ # Data directories
│ ├── raw/ # Original datasets
│ └── processed/ # Cleaned and transformed data
│
├── notebooks/ # Jupyter notebooks
│ ├── 01_data_understanding.ipynb
│ ├── 02_data_cleaning.ipynb
│ ├── 03_data_transformation.ipynb
│ ├── 04_data_reduction.ipynb
│ └── 05_data_imbalance_handling.ipynb
│
├── src/ # Source code
│ ├── data_understanding.py # Phase 1: Data assessment
│ ├── data_cleaning.py # Phase 2: Data cleaning
│ ├── data_transformation.py # Phase 3: Feature engineering
│ ├── data_reduction.py # Phase 4: Feature selection
│ └── data_imbalance.py # Phase 5: Imbalance handling
│
├── reports/ # Documentation
│ └── Diabetes_Data_Preparation_Report.md
│
└── README.md

## 🔄 5-Phase Data Preparation Pipeline

### Phase 1: Data Understanding & Quality Assessment
- Comprehensive data profiling and statistical analysis
- Biological impossibility detection (zeros in medical measurements)
- Missing data pattern analysis
- Initial data quality assessment report

### Phase 2: Data Cleaning & Imputation
- Replacement of biologically impossible zeros with NaN
- Medical range validation and outlier detection
- Robust median imputation for skewed distributions
- Data quality improvement and validation

### Phase 3: Data Transformation & Feature Engineering
- Clinical category creation (Age groups, BMI categories, Glucose categories)
- Feature scaling and encoding strategies
- Interaction feature engineering (Metabolic_Score, BMI_Age_Interaction)
- Dataset expansion from 9 to 32 features

### Phase 4: Data Reduction & Feature Selection
- Multi-method feature selection (5 different approaches)
- PCA analysis for dimensionality reduction
- Consensus feature selection methodology
- Optimal feature set identification (15 features)

### Phase 5: Data Imbalance Handling
- Comprehensive class distribution analysis
- Multiple sampling technique evaluation
- Optimal method selection (Random Oversampling)
- Balanced dataset creation with improved sensitivity

## 📈 Key Results

### Data Quality Improvements
- **Missing Data Resolution**: 652 values imputed across 5 biological features
- **Medical Validity**: All values within clinically reasonable ranges
- **Feature Relevance**: 15 optimally selected clinically meaningful features

### Performance Enhancements
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Sensitivity | 53.7% | 70.4% | **+31.1%** |
| AUC-ROC | 0.832 | 0.834 | +0.002 |
| Specificity | 86.0% | 80.0% | -6.0% (acceptable) |

### Feature Engineering Success
- **Metabolic_Score**: Emerged as strongest predictor
- **Clinical Categories**: Enhanced model interpretability
- **Interaction Features**: Captured complex biological relationships

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook
