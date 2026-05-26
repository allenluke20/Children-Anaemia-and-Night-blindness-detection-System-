# Nutritional Deficiency Disease Classification

A machine learning project that predicts nutritional deficiency diseases based on patient demographics, lifestyle factors, dietary habits, clinical symptoms, and laboratory values.

## Overview

This project builds a multi-class classification pipeline to diagnose five health conditions:
- **Healthy**
- **Anemia**
- **Night Blindness**
- **Rickets / Osteomalacia**
- **Scurvy**

Using a dataset of 4,000 patient records with 29 features, the pipeline preprocesses mixed data types and trains multiple supervised learning models. A Decision Tree Classifier is selected as the final model after hyperparameter tuning, achieving **99%+ accuracy** with only **3 misclassifications** on the held-out test set.

## Repository Structure

```
├── data/
│   └── vitamin_deficiency_disease_dataset_cleaned.csv          # Raw dataset (4,000 records)
├── notebooks/
│   └── Vitamin deficiency classifier.ipynb        # Full analysis & modeling pipeline
├── models/
│   └── model.pkl      # Serialized final model
├── README.md
├── requirements.txt
├── LICENSE
├── CONTRIBUTING.md
└── .gitignore
```

## Dataset

- **Size:** 4,000 rows × 30 columns
- **Target:** `disease_diagnosis` (5 classes)
- **Features:** 29 predictive variables covering:
  - **Demographics:** `age`, `gender`, `bmi`
  - **Lifestyle:** `smoking_status`, `alcohol_consumption`, `exercise_level`, `sun_exposure`
  - **Diet:** `diet_type` (Omnivore, Vegetarian, Vegan, Pescatarian), vitamin intake (% RDA), mineral intake (% RDA)
  - **Clinical Labs:** `hemoglobin_g_dl`, `serum_vitamin_d_ng_ml`, `serum_vitamin_b12_pg_ml`, `serum_folate_ng_ml`
  - **Symptoms (binary):** night blindness, fatigue, bleeding gums, bone pain, muscle weakness, numbness/tingling, memory problems, pale skin, dry skin

- **Data Quality:** No missing values; clean, structured tabular data.

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Distribution analysis (e.g., BMI by gender)
- Correlation heatmap of numerical features
- Class distribution review

### 2. Preprocessing Pipeline
| Step | Categorical Features | Numerical Features |
|------|---------------------|-------------------|
| Encoding | `OneHotEncoder` (dense output) | — |
| Scaling | `StandardScaler` | `StandardScaler` |
| Tooling | `ColumnTransformer` via `make_column_transformer` | |

- **Categorical columns:** `gender`, `smoking_status`, `alcohol_consumption`, `exercise_level`, `diet_type`, `sun_exposure`
- **Numerical columns:** all remaining 23 features

### 3. Model Benchmarking (3-Fold Cross-Validation)

| Model | Accuracy | Macro Avg F1 | Notes |
|-------|----------|--------------|-------|
| Logistic Regression | 91% | 0.79 | Baseline; struggles with minority classes |
| **Decision Tree** | **99%** | **0.97** | **Best overall balance; selected for tuning** |
| K-Nearest Neighbors | 74% | 0.55 | Poor performance on imbalanced classes |
| Random Forest | 97% | 0.89 | Strong, but slightly lower recall on Scurvy |

### 4. Hyperparameter Tuning
- **Technique:** `GridSearchCV` (3-fold CV)
- **Estimator:** Pipeline(DecisionTreeClassifier)
- **Grid:**
  - `criterion`: `['gini', 'entropy']`
  - `max_depth`: `[None, 3, 5, 7, 10]`
  - `min_samples_split`: `[2, 5, 10]`
  - `min_samples_leaf`: `[1, 2, 4]`
  - `class_weight`: `[None, 'balanced']`
- **Best Result:** ~99.4% mean CV accuracy with balanced class weights and `entropy` criterion.

### 5. Final Model
- Retrained the best estimator on the **full dataset** (X, y)
- Evaluated on 20% held-out test set (stratified split, `random_state=42`)
- **Test Set Performance:** 799/800 correct — **only 1 misclassification**

## Key Results

- The Decision Tree Classifier perfectly captures symptom-disease relationships (e.g., bleeding gums → Scurvy, bone pain → Rickets, pale skin/fatigue → Anemia).
- Feature engineering was minimal; raw clinical and dietary inputs proved highly predictive.
- Class balancing (`class_weight='balanced'`) provided marginal but consistent gains on minority classes (Night Blindness, Scurvy).

## Installation

```bash
git clone https://github.com/yourusername/nutritional-deficiency-classification.git
cd nutritional-deficiency-classification
pip install -r requirements.txt
```

## Usage

```python
import joblib
import pandas as pd

# Load model
model = joblib.load('models/best_decision_tree.pkl')

# Prepare patient data (same 29 columns, same order)
patient = pd.DataFrame([{
    'age': 34,
    'gender': 'Female',
    'bmi': 22.5,
    'smoking_status': 'Never',
    'alcohol_consumption': 'Light',
    'exercise_level': 'Active',
    'diet_type': 'Vegetarian',
    'sun_exposure': 'Moderate',
    'vitamin_a_percent_rda': 85.0,
    'vitamin_c_percent_rda': 60.0,
    # ... remaining features
}])

# Predict
prediction = model.predict(patient)
print(f"Diagnosis: {prediction[0]}")
```

## Limitations & Next Steps

- **Dataset:** Synthetic/simulated medical data; real-world deployment requires validation on clinical cohorts.
- **Generalization:** Decision trees can overfit; Random Forest or Gradient Boosting may offer more robustness at scale.
- **Interpretability:** SHAP or feature-importance analysis would strengthen clinical trust.
- **Imbalance:** Night Blindness and Scurvy are rare (~2% each); SMOTE or class-specific thresholds could be explored.

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Author

Allen Lukelo Mahobe — 0777928293
