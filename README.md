# Wine Analysis Project

## Overview

This project focuses on analyzing and predicting wine quality using machine learning techniques. It employs decision tree models (CART) and feedforward neural networks (FNN), with and without oversampling techniques like SMOTE. The goal is to classify wine quality based on physicochemical properties and improve model performance through advanced methods.

---

## Features

### 1. **Datasets**
- **Red Wine Dataset**: Contains physicochemical properties and quality ratings for red wines.
- **White Wine Dataset**: Contains similar data for white wines.
- Both datasets are combined for analysis, with an added `wine_type` column to distinguish between red (`0`) and white (`1`) wines.

### 2. **Data Preprocessing**
- Duplicate rows removed.
- Missing values checked.
- Data merged into a single dataset with appropriate factorization.
- Features scaled using Min-Max normalization for neural network models.

### 3. **Exploratory Data Analysis (EDA)**
- Boxplots comparing residual sugar, free sulfur dioxide, and total sulfur dioxide across wine types.
- Histograms showing the distribution of features.
- Correlation heatmap to identify relationships between variables.
- Analysis of alcohol content and its relationship with wine quality.

### 4. **Machine Learning Models**
#### **Classification And Regression Tree (CART)**
- Decision tree models trained on wine data.
- Pruned models based on cross-validation error for optimal performance.

#### **Oversampling Techniques**
- SMOTE: Synthetic Minority Oversampling Technique applied to balance class distribution.

#### **Feedforward Neural Network (FNN)**
- Built using the `torch` library in R.
- Multi-layer architecture with ReLU activation functions and Adam optimizer.

---

## Results

The table below summarizes the performance of various models tested in this project:

| Model Type                     | Accuracy | Precision | Recall | F1 Score | Specificity |
|--------------------------------|----------|-----------|--------|----------|-------------|
| Decision Tree (w/o SMOTE)      | 0.5715   | 0.5500    | 0.5300 | 0.5324   | 0.7672      |
| Decision Tree (SMOTE)          | 0.5672   | 0.5642    | 0.5755 | 0.5676   | 0.7830      |
| Feedforward NN (w/o SMOTE)     | 0.6168   | 0.6137    | 0.5823 | 0.5932   | 0.7903      |
| Feedforward NN (SMOTE)         | 0.5949   | 0.6045    | 0.5994 | 0.6007   | 0.7951      |

---

## Dependencies

### Required R Libraries:
```R
install.packages(c("ggplot2", "dplyr", "corrplot", "rpart", "caret", "ggparty", "torch", "smotefamily"))
```

---

## How to Run

### Steps:
1. Clone or download the project repository.
2. Place the wine datasets (`winequality-red.csv` and `winequality-white.csv`) in the appropriate directory specified in the script.
3. Install required libraries using the command above.
4. Run the script `Red_white_analysis.R` in RStudio or any R IDE.

---

## Outputs

### Visualizations:
- Boxplots comparing wine properties across types.
- Histograms showing feature distributions.
- Correlation heatmap of wine attributes.
- Barplots showing average quality by alcohol levels.

### Model Results:
- Confusion matrices for CART and FNN models (including oversampled versions).
- Performance metrics such as accuracy, precision, recall, F1-score, and specificity.

---



## Contact
For questions or suggestions regarding this project, feel free to reach out!
