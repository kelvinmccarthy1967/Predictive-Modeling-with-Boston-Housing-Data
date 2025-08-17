# 🏡 Boston Housing Predictive Modeling  

## 📌 Overview  
This project explores the **Boston Housing dataset** to analyze the factors that influence housing prices and to build predictive models. Both **regression** and **classification** approaches were applied, enabling comparison across different machine learning techniques.  

The project demonstrates **statistical analysis, model selection, and predictive modeling** skills essential for real-world data science applications.  

---

## 🔍 Key Steps  
- **Data Exploration & Preprocessing**  
  - Summary statistics, missing value checks, and outlier detection  
  - Correlation analysis and visualizations (heatmaps, scatterplots)  

- **Regression Modeling**  
  - Linear Regression with `crim ~ tax`  
  - Stepwise selection using **AIC**  
  - Ridge & Lasso regression for regularization  
  - Confidence and prediction intervals  

- **Classification Modeling**  
  - GLM (logistic regression)  
  - Decision Tree classifier for predicting `crimbin`  

- **Advanced Models**  
  - Poisson regression (GLM) for `rad`  
  - Random Forest for regression  

- **Model Evaluation**  
  - Metrics: **MAE, RMSE, R², Accuracy, AUC**  
  - Cross-validation for robustness  

---

## 📊 Results  
- **Regression:** Ridge & Lasso improved generalization compared to OLS.  
- **Classification:** Random Forest outperformed GLM and Decision Tree in predicting categorical outcomes.  
- **Feature Insights:** Tax rates, pupil-teacher ratio, and accessibility to highways were strong predictors.  

---

## ⚙️ Tools & Libraries  
- Python, Jupyter Notebook  
- **Pandas, NumPy** for data wrangling  
- **Matplotlib, Seaborn** for visualization  
- **Scikit-learn, Statsmodels** for modeling  
- **Random Forest, Ridge, Lasso, GLM**  


