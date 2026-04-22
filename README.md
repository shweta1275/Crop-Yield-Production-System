# Crop Yield Production System

## 1. Project Overview
This repository implements an advanced machine learning pipeline and interactive web application for crop yield prediction. By integrating robust preprocessing, dimensionality reduction, unsupervised clustering, and multi-model regression architectures, the system captures complex spatial and environmental interactions to forecast agricultural output with high precision. The project features a modern, polished Streamlit frontend for real-time, user-friendly predictions.

## 2. Problem Statement
Accurate crop yield estimation is critical for optimizing agricultural resource allocation and ensuring food security. The objective is to develop a highly predictive regression system that estimates `Crop Yield (kg per hectare)` by modeling complex, non-linear dependencies among diverse agronomic and environmental variables.

## 3. Dataset Description
The system processes `enhanced_crop_yield_dataset.csv`, a multidimensional dataset comprising continuous and categorical agricultural covariates. The continuous target variable is `Crop Yield (kg per hectare)`. The data encompasses intricate feature interactions that require advanced feature engineering to isolate predictive signals.

## 4. Methodology
The methodology reflects a structured and intentional machine learning pipeline designed to maximize predictive capacity:
- **Encoding & Standardization:** Categorical features undergo one-hot encoding with collinearity reduction. Continuous variables are normalized via `StandardScaler` to ensure uniform gradient propagation and accurate distance calculations.
- **Dimensionality Reduction (PCA):** Principal Component Analysis (PCA) is applied to retain 95% of the explained variance. This mitigates the curse of dimensionality, filters intrinsic noise, and accelerates model convergence.
- **Clustering-Based Feature Engineering (DBSCAN):** Density-Based Spatial Clustering of Applications with Noise (DBSCAN) identifies non-linear, localized structures within the principal components. These cluster assignments are injected as a latent spatial feature, enhancing the models' contextual awareness of underlying data topologies.
- **Pipeline Architecture:** The augmented feature matrix is systematically partitioned into an 80/20 train-test split to facilitate rigorous, out-of-sample validation.

## 5. Models Used
The system evaluates a progression of models to benchmark performance:
- **Linear Regression:** Serves as the statistical baseline, modeling the fundamental linear relationships between the engineered features and agricultural yield.
- **Decision Tree Regressor:** Captures hierarchical, non-linear feature interactions and threshold-based dependencies that linear combinations fail to resolve.
- **Deep Neural Network (TensorFlow/Keras):** A multi-layer sequential architecture designed to model highly complex function approximations. The network leverages dense representations to extract deep feature interactions, significantly increasing predictive capacity.

## 6. Evaluation Metrics
Model performance is quantified using robust regression metrics:
- **Root Mean Squared Error (RMSE):** Selected to heavily penalize large prediction deviations, providing a strict measure of the expected error magnitude in the yield forecasts.
- **R-squared ($R^2$):** Quantifies the proportion of yield variance successfully captured by the models, serving as the primary indicator of overall goodness-of-fit.

## 7. Results
The comparative evaluation demonstrates that the Deep Neural Network achieves the highest predictive accuracy. While Linear Regression establishes a reliable baseline, its performance is fundamentally constrained by the non-linear nature of agronomic variables. The Decision Tree successfully maps threshold effects, but the Neural Network consistently yields the lowest RMSE and highest $R^2$. The neural architecture's superior performance is directly attributed to its capacity to approximate high-dimensional, complex non-linear functions and fully leverage the dense, structural features extracted by the PCA and DBSCAN pipeline.

## 8. How to Run
1. Ensure a Python 3.x environment is established.
2. Install the required pipeline dependencies using the provided requirements file:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `enhanced_crop_yield_dataset.csv` in the project root directory.

### Running the Interactive Frontend (Recommended)
To launch the modern, card-based web interface:
```bash
streamlit run app.py
```
This will automatically open the application in your default web browser (typically at `http://localhost:8501`).

### Running the Terminal Pipeline
To execute the raw machine learning pipeline script via terminal:
```bash
python model.py
```

## 9. Technologies Used
- **Core Processing:** Python, pandas, NumPy
- **Machine Learning & Feature Engineering:** scikit-learn (StandardScaler, PCA, DBSCAN, Linear Regression, Decision Tree, Metrics)
- **Deep Learning Architecture:** TensorFlow, Keras (Sequential API)
- **Frontend / UI:** Streamlit (Interactive Web Application)
