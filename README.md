# AI-Powered Vitamin Deficiency Screening for Children

> **Capstone II Project** — Early Detection of Anaemia and Night Blindness in Tanzania and East Africa

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)](https://xgboost.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Model Performance](#model-performance)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Model](#training-the-model)
  - [Running the Streamlit App](#running-the-streamlit-app)
- [Clinical Recommendations](#clinical-recommendations)
- [Project Structure](#project-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Acknowledgments](#acknowledgments)

---

## 🌍 Project Overview

This project develops a machine learning system to screen pediatric patients for vitamin deficiencies—specifically **Anaemia** (iron deficiency) and **Night Blindness** (Vitamin A deficiency)—using readily available clinical and demographic data. Built as a Capstone II project, the system is designed for deployment in rural Tanzanian clinics where laboratory infrastructure is limited and Community Health Workers (CHWs) need decision-support tools to triage high-risk children.

**Key Innovation:** A lightweight, interpretable XGBoost classifier that runs on low-cost Android tablets, turning a simple symptom checklist and a finger-prick hemoglobin reading into an instant triage score with WHO-aligned clinical actions.

---

## 🩺 Problem Statement

### The Crisis in East Africa

- **58%** of children under 5 in Tanzania suffer from anemia (WHO, 2023)
- **~33%** of preschool children are affected by Vitamin A deficiency
- **70%** of rural clinics lack hemoglobin laboratory infrastructure
- Delayed diagnosis leads to **irreversible developmental impairment**, blindness, and mortality

### The Gap

Community Health Workers in rural Tanzania serve as the frontline of pediatric care, yet they lack diagnostic tools to identify micronutrient deficiencies early. District hospitals are overwhelmed, and confirmatory lab tests cost **$8–12 per child**—prohibitive at scale. There is an urgent need for a **low-cost, rapid triage tool** that can identify high-risk children before complications become severe.

---

## 📊 Dataset

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Integer | Patient age in years (1–17) |
| `gender` | Categorical | Male / Female |
| `bmi` | Float | Body Mass Index |
| `diet_type` | Categorical | Omnivorous / Vegetarian |
| `hemoglobin_g_dl` | Float | Hemoglobin concentration (g/dL) |
| `has_night_blindness` | Binary | Night blindness symptom (0/1) |
| `has_fatigue` | Binary | Fatigue symptom (0/1) |
| `has_pale_skin` | Binary | Pale skin symptom (0/1) |
| `has_dry_skin` | Binary | Dry skin symptom (0/1) |
| `other_diagnosis` | Categorical | Malaria / Sickle cell / No other disease |
| `disease_diagnosis` | Target | **Healthy** / **Anaemia** / **Night_blindness** |

- **Size:** 10,000 pediatric records
- **Source:** Simulated dataset calibrated to Tanzanian epidemiology
- **Class Distribution:** Healthy (57%), Anaemia (15%), Night Blindness (8%)
- **Key Insight:** Females show 2× higher Anaemia prevalence; hemoglobin centers at 12.8 g/dL (below WHO normal threshold of 11.5 g/dL for children)

---

## ⚙️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT LAYER                              │
│  Raw patient record (age, gender, BMI, diet, Hb, symptoms)   │
└──────────────────┬──────────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────────┐
│              PREPROCESSING PIPELINE                         │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ Categorical Pipeline │  │   Numerical Pipeline        │  │
│  │ OneHotEncoder        │  │   PowerTransformer          │  │
│  │ PowerTransformer     │  │   (Yeo-Johnson)             │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│              │                          │                    │
│              └──────────┬───────────────┘                    │
│                         │                                    │
│              ┌──────────▼───────────────┐                    │
│              │  ColumnTransformer       │                    │
│              └──────────┬───────────────┘                    │
└─────────────────────────┼──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│                 MODEL LAYER                                 │
│              XGBoost Classifier                             │
│         (multi:softprob objective)                          │
│                                                             │
│  Hyperparameters (tuned via GridSearchCV):                  │
│  • max_depth: 7                                             │
│  • learning_rate: 0.1                                       │
│  • n_estimators: 100                                        │
│  • scale_pos_weight: class-balanced                         │
└─────────────────────────┬──────────────────────────────────┘
                          │
┌─────────────────────────▼──────────────────────────────────┐
│                OUTPUT LAYER                                 │
│  Predicted Class + Probability Distribution                 │
│  → Clinical Recommendation Engine (Streamlit)                │
└─────────────────────────────────────────────────────────────┘
```

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Python, pandas, NumPy |
| Preprocessing | scikit-learn (ColumnTransformer, Pipeline, OneHotEncoder, PowerTransformer) |
| Modeling | XGBoost, scikit-learn |
| Hyperparameter Tuning | GridSearchCV (3-fold stratified cross-validation) |
| Visualization | Seaborn, Matplotlib |
| Deployment | Streamlit, joblib |

---

## 📈 Model Performance

### Cross-Validation Results (5 Models Evaluated)

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 94.0% | 0.91 | Weak on minority classes |
| Decision Tree | 99.9% | 0.99 | Overfitting risk |
| K-Nearest Neighbors | 96.0% | 0.94 | Sensitive to symptom sparsity |
| Random Forest | 99.9% | 1.00 | Excellent, close second |
| **XGBoost (Final)** | **99.8%** | **0.998** | **Best generalization** |

### Final Test Set Performance (n = 2,000 held-out records)

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | **99.65%** |
| **Macro F1-Score** | **0.997** |
| **Weighted F1-Score** | **0.997** |

| Class | Precision | Recall | F1-Score | Correct / Total |
|-------|-----------|--------|----------|-----------------|
| Anaemia | 0.99 | 0.99 | 0.99 | 372 / 374 |
| Healthy | 1.00 | 1.00 | 1.00 | 1,431 / 1,435 |
| Night Blindness | 1.00 | 0.99 | 1.00 | 190 / 191 |

**Total Misclassifications:** Only 7 out of 2,000 cases (0.35%)

### Feature Importance

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `has_night_blindness` | 49.6% |
| 2 | `has_fatigue` | 10.1% |
| 3 | `diet_type_omnivorous` | 9.7% |
| 4 | `has_pale_skin` | 8.4% |
| 5 | `hemoglobin_g_dl` | 4.7% |
| 6 | `gender_Female` | 4.6% |

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vitamin-deficiency-classifier.git
cd vitamin-deficiency-classifier

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.3.0
xgboost>=2.0.0
seaborn>=0.12.0
matplotlib>=3.6.0
streamlit>=1.28.0
joblib>=1.2.0
```

---

## 💻 Usage

### Training the Model

```bash
# Run the training notebook or script
jupyter notebook Vitamin_deficiency_classifier.ipynb

# Or execute the Python script
python train_model.py
```

This will:
1. Load and preprocess the dataset
2. Train 5 candidate models with cross-validation
3. Run GridSearchCV hyperparameter tuning on XGBoost
4. Export the final model as `model.pkl` and class labels as `classes.pkl`

### Running the Streamlit App

```bash
# Launch the clinical decision support interface
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`. Enter patient details to receive:
- Instant prediction (Anaemia / Healthy / Night Blindness)
- Confidence score
- **Clinical action checklist** tailored to the prediction

---

## 🏥 Clinical Recommendations

The system includes a **clinical recommendation engine** that translates AI predictions into actionable, WHO-aligned treatment protocols for Community Health Workers.

### If Predicted: Anaemia

1. **Immediate iron-folate supplementation** — WHO standard pediatric dosing (2 mg/kg elemental iron daily)
2. **Counsel caregiver on iron-rich foods** — leafy greens (mchicha), beans, lentils, liver, red meat
3. **Schedule follow-up hemoglobin test** — re-check in 8 weeks to monitor response
4. **Flag for malaria co-treatment** — current malaria episode may worsen anemia; ensure full ACT course
5. **Refer to district hospital urgently if:**
   - Hemoglobin < 8 g/dL (severe anemia threshold)
   - Signs of heart failure (tachycardia, gallop rhythm, hepatomegaly)
   - Pallor + respiratory distress
6. **Deworming** — administer albendazole if not given in last 6 months

### If Predicted: Night Blindness

1. **Immediate high-dose Vitamin A** — WHO protocol: 200,000 IU oral (100,000 IU if age < 12 months)
2. **Dark adaptation test referral** — schedule at district eye clinic within 2 weeks
3. **Counsel on Vitamin A-rich diet** — pumpkin, carrots, sweet potato, spinach, eggs, liver, palm oil
4. **Deworming** — administer albendazole if not done in last 6 months (helminths worsen VAD absorption)
5. **Re-screen in 3 months** — check for symptom resolution and serum retinol if available
6. **Refer to ophthalmology if:**
   - No improvement after 3 months of supplementation
   - Signs of corneal xerosis or Bitot spots (advanced VAD)
   - Bilateral blindness or visual impairment

### If Predicted: Healthy

1. Continue balanced diet with iron and Vitamin A-rich foods
2. Routine growth monitoring at monthly CHW visits
3. Verify up-to-date immunizations per EPI calendar
4. Deworming every 6 months per national guidelines
5. Schedule next routine screening in 6 months

> ⚠️ **Disclaimer:** This AI tool supports—but does not replace—clinical judgment. Confirm all high-risk predictions with laboratory testing and physician review.

---

## 📁 Project Structure

```
vitamin-deficiency-classifier/
│
├── data/
│   └── children_1.csv                    # Pediatric health dataset (10,000 records)
│
├── notebooks/
│   └── Vitamin_deficiency_classifier.ipynb   # Full analysis & model training
│
├── models/
│   ├── model.pkl                         # Trained XGBoost pipeline
│   └── classes.pkl                       # Class label encoder
│
├── src/
│   ├── train_model.py                    # Training script
│   ├── clinical_recommendation.py        # Streamlit recommendation component
│   └── utils.py                          # Helper functions
│
├── app.py                                # Streamlit application entry point
│
├── requirements.txt                      # Python dependencies
│
├── README.md                             # This file
│
└── presentation/
    └── Capstone_Vitamin_Deficiency_Presentation.pptx
```

---

## ⚠️ Limitations

1. **Simulated Data:** Records are calibrated to Tanzanian epidemiology but not sourced from live EMRs. Real-world clinical validation is required before deployment.
2. **Binary Symptom Simplification:** Real clinical presentation is graded (mild/moderate/severe), not 0/1. This may miss subclinical cases.
3. **Age Restriction:** Trained only on ages 1–17. Not applicable to infants <1 year or adults.
4. **Hemoglobin Dependency:** Still requires a finger-prick hemoglobin reading. Does not fully replace laboratory diagnostics, but accelerates triage.
5. **Clinical Cost of Errors:** Though rare (0.35%), misclassifying Anaemia as Healthy carries higher stakes than the reverse. Continuous monitoring and confirmatory testing are essential.

---

## 🔮 Future Work

- **Phase 1:** Pilot in 5 Dodoma district clinics (Q1 2027) with independent clinical audit
- **Phase 2:** Integration with DHIS2 (Tanzania's national health data platform)
- **Phase 3:** Open-source expansion to Kenya, Uganda, and Rwanda
- **Model Improvements:** Incorporate graded symptom severity, serum retinol levels, and seasonal variation
- **Hardware:** Optimize for offline deployment on $50 Android tablets using TensorFlow Lite

---

## 🙏 Acknowledgments

- **Capstone Advisors** for guidance on clinical relevance and model interpretability
- **Tanzania Ministry of Health** for epidemiological context and DHIS2 integration roadmap
- **PATH / D-tree International** for insights on Community Health Worker workflows
- **WHO** for pediatric anemia and Vitamin A deficiency treatment protocols
- **scikit-learn & XGBoost communities** for open-source machine learning tools

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 📧 Contact

For questions, collaboration, or deployment inquiries:

- **Project Lead:** [Your Name] — your.email@university.edu
- **GitHub Issues:** [github.com/yourusername/vitamin-deficiency-classifier/issues](https://github.com/yourusername/vitamin-deficiency-classifier/issues)

---

> *"Technology is most powerful when it empowers the people on the front lines."* — Built for Tanzania's Community Health Workers.
