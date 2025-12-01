# Benchmarking AlphaFold3-like Methods for Protein-Peptide Complex Prediction
This repository contains the code and datasets supporting the paper "Benchmarking AlphaFold3-like Methods for Protein-Peptide Complex Prediction". The materials herein are designed to facilitate reproducibility of the machine learning-based benchmarking analyses presented in the study.


## Repository Structure

| File/Directory               | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `ML_model_1.py`              | Implementation of machine learning models (excluding AutoGluon) used for benchmarking. Includes regression models, hyperparameter tuning, and evaluation pipelines. |
| `ProteinPeptide_ML.csv`      | Core dataset containing protein-peptide complex features and target variables for model training/evaluation. |
| `dataset1.csv` - `dataset4.csv` | Datasets used in the study, including variant subsets and validation cohorts. |
||

## Machine Learning Models

### Included in `ML_model_1.py`
This script implements a suite of regression algorithms for predicting protein-peptide complex properties, including:
- Linear models: Linear Regression , Ridge Regression, Lasso Regression, ElasticNet Regression
- Tree-based models: Decision Tree, Random Forest, Gradient Boosting, Bagging Regressor, XGBoost
- Distance-based model: K-Nearest Neighbors Regression (KNN)
- Kernel-based model: Support Vector Regression (SVR)

#### 1. Read and Preprocess Data
- Loads data from `ProteinPeptide_ML.csv`.
- Identifies X features and Y (target variable).
- Fills missing values with column means.
- Performs MinMax normalization.
- Extracts name and method fields.

#### 2. Split into Train/Test Sets by Unique Protein Name
Instead of random row-wise splitting (which would cause data leakage), the script:
- Extracts unique protein names.
- Randomly assigns 80% of names to the training set and 20% to the test set.
- Filters rows accordingly.

This ensures the model never sees the same protein name in both train and test sets.

#### 3. Train Multiple Regression Models (with Hyperparameter Tuning)
The script creates a dictionary of models, each wrapped in `GridSearchCV` with cross-validation:
- Random Forest Regressor
- Linear Regression
- Support Vector Regression (SVR)
- XGBoost Regressor
- Decision Tree Regressor
- Ridge Regression
- Lasso Regression
- ElasticNet
- KNN Regressor
- Gradient Boosting Regressor
- Bagging Regressor

#### 4. Unified Function to Evaluate Each Model
For each model, the script:
- Fits the model on the training data.
- Predicts on the test set.
- Computes evaluation metrics:
  - Pearson correlation
  - Mean Squared Error (MSE)
  - R² score
- Saves outputs:
  - Evaluation results in a text file.
  - The trained model as a `.pkl` file.
  - A scatter plot of True vs Predicted values.
- Adds predictions to `test_data` (as new columns).

#### 5. Top-N (Top-3) Prediction Ranking Evaluation
For each model, and for each protein name and method:
1. Select the rows belonging to that name & method.
2. Pick the top-3 samples with the highest predicted values.
3. Check whether `(predicted > 0.8)` matches `(true value > 0.8)`.
4. Count correct vs total predictions.

#### 6. Requirements

- pip install numpy pandas matplotlib seaborn scipy joblib
- pip install scikit-learn xgboost
- PyTorch (choose based on CUDA version; install CPU version if no GPU)
  - CPU version
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  - GPU version (match CUDA version; example for CUDA 12.1)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

### AutoGluon Models
AutoGluon-based experiments are maintained in a separate repository:  
[https://github.com/mahofai/protein_peptide_autogluon](https://github.com/mahofai/protein_peptide_autogluon)  

This repository contains code for automated machine learning (AutoML) workflows using AutoGluon, with comparative analyses against the models in `ML_model_1.py`.