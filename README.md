📌 Crop Yield Prediction using Machine Learning

📖 Overview

This project predicts crop yield (kg per hectare) using machine learning techniques based on environmental, soil, and agricultural factors. Multiple models are implemented and compared to analyze performance.

🎯 Objective
Predict crop yield using real-world features
Compare different ML models
Identify the best-performing approach
🧠 Models Used
Linear Regression (baseline)
Decision Tree Regressor
Neural Network (TensorFlow/Keras)
⚙️ Methodology
Data preprocessing and cleaning
One-hot encoding of categorical features
Feature scaling using StandardScaler
Dimensionality reduction using PCA
Model training and evaluation
Performance comparison using RMSE and R² score
📊 Evaluation Metrics
RMSE (Root Mean Squared Error) → prediction error
R² Score → model accuracy
📈 Results Summary
Linear Regression → low performance
Decision Tree → moderate improvement
Neural Network → best performance (captures nonlinear patterns)
🛠️ Requirements

Install dependencies:

pip install pandas numpy scikit-learn matplotlib seaborn tensorflow-macos tensorflow-metal

(For Mac M-series with Python ≤3.11)

▶️ How to Run
python model.py
📁 Project Structure
MINIPROJECT/
│── model.py
│── enhanced_crop_yield_dataset.csv
│── README.md
💡 Key Insight

Agricultural data contains complex nonlinear relationships, which are better captured by neural networks compared to traditional models.
