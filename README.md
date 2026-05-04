# Virulence-Protein-Predictor

### A Computational Pipeline for Sequence-Based Virulence Classification

---

## Abstract

Accurate identification of virulent proteins is critical for understanding pathogenic mechanisms and enabling therapeutic target discovery.
This project presents a **machine learning-based computational framework** that classifies proteins as *virulent* or *non-virulent* using sequence-derived features.

A comprehensive feature set (>500 features) was engineered from protein sequences, followed by training and evaluation of multiple machine learning models. The pipeline demonstrates robust predictive performance and provides an extensible framework for bioinformatics-driven pathogen analysis.

---

## Objectives

* Develop a **feature-rich representation** of protein sequences
* Build and compare multiple ML models for virulence prediction
* Evaluate performance using robust statistical metrics
* Enable prediction on unseen protein sequences
* Provide a **fully reproducible pipeline**

---

## Methodology

### 1. Feature Engineering

Protein sequences were transformed into numerical representations using:

* **Amino Acid Composition (AAC)**
* **Dipeptide Composition (DPC)**
* **Physicochemical properties** (molecular weight, GRAVY, instability index)
* **Charge & polarity-based features**
* **Signal peptide and structural heuristics**

Total features: **500+ per sequence**

---

### 2. Data Preprocessing

* Train–Validation–Test split
* Standardization (scaling)
* Class imbalance handling using **SMOTE**

---

### 3. Model Development

The following classifiers were implemented:

* Random Forest
* XGBoost
* Support Vector Machine (RBF kernel)
* Logistic Regression

---

### 4. Evaluation Strategy

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)

Additionally:

* ROC and Precision-Recall curves
* Confusion matrices
* Cross-validation and enhanced validation analysis

---

### 5. Prediction Framework

A prediction module enables:

* Input: FASTA file
* Output:

  * Class label (Virulent / Non-virulent)
  * Probability score
  * Ensemble prediction (majority voting)

---

## Project Structure

```bash
virulence-protein-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/
│   └── processed/
│
├── src/
│   ├── feature_extraction.py
│   ├── preprocess.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   ├── predict.py
│   ├── feature_extractor.py
│   ├── validate_models.py
│   └── validate_models_enhanced.py
│
├── pipeline/
│   └── master_pipeline.py
│
├── models/
│
├── results/
│   ├── figures/
│   └── validation/
│
└── examples/
    └── sample.fasta
```

---

## Reproducibility

### Installation

```bash
git clone https://github.com/your-username/virulence-protein-prediction.git
cd virulence-protein-prediction
pip install -r requirements.txt
```

### Run Full Pipeline

```bash
python pipeline/master_pipeline.py
```

---

## Predict on New Sequences

```bash
python src/predict.py --input examples/sample.fasta
```

---

## Results (Summary)

* Multi-model evaluation demonstrated strong classification performance
* Ensemble predictions improved robustness
* Feature-rich representation significantly enhanced model accuracy

---

## Scientific Significance

* Enables **high-throughput virulence prediction**
* Supports **drug target identification and pathogen analysis**
* Demonstrates the power of **sequence-based ML in computational biology**
* Provides a **scalable and extensible bioinformatics framework**

---

## Limitations

* Relies solely on sequence-derived features
* No structural or experimental validation included
* Performance depends on dataset quality and class balance

---

## Future Work

* Integration of structural features (AlphaFold / PDB)
* Deep learning architectures (CNN/RNN/Transformers)
* External validation on independent datasets
* Web server or API deployment

---

## Author

**Vanathi Shanmugam**  
Bioinformatics | Genomics | Machine Learning  

🔗 LinkedIn: www.linkedin.com/in/vanathi-shanmugam-26127928a

---

## License

This project is intended for academic and research purposes.
