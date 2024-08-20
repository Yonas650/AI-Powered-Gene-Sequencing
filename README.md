
# AI-Powered Gene Sequencing

## Project Overview

This project aims to classify different types of cancer based on gene expression profiles using various machine learning models, including Random Forest, Support Vector Machine (SVM), and Convolutional Neural Networks (CNN). The dataset used in this project is the **BRCA Multi-Omics (TCGA)** dataset, which contains gene expression data and associated clinical information.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Development](#model-development)
  - [Random Forest](#random-forest)
  - [Support Vector Machine (SVM)](#support-vector-machine-svm)
  - [Convolutional Neural Network (CNN)](#convolutional-neural-network-cnn)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Future Work](#future-work)
- [Requirements](#requirements)
- [How to Run the Project](#how-to-run-the-project)

## Dataset

- **Name:** BRCA Multi-Omics (TCGA)
- **Source:** [Kaggle - BRCA Multi-Omics (TCGA)](https://www.kaggle.com/datasets/samdemharter/brca-multi-omics-tcga)
- **Files:**
  - `brca_data_w_subtypes.csv`: Contains gene expression data and associated clinical labels, including histological types of cancer.
  - `data.csv`: General data used for supplementary analysis.

## Preprocessing

### Steps:
1. **Loading Data:** The gene expression data is loaded from `brca_data_w_subtypes.csv`.
2. **Handling Missing Values:** Missing values in numeric columns are imputed using the median, while categorical columns are imputed using the most frequent value.
3. **Feature Scaling:** Features are standardized using `StandardScaler` to have zero mean and unit variance.
4. **Feature Selection:** Features with low variance are removed using a variance threshold.
5. **Label Encoding:** Categorical labels (histological types) are encoded into integers for compatibility with machine learning models.
6. **Data Splitting:** The dataset is split into training and testing sets with an 80-20 split.

## Model Development

### Random Forest

- **Description:** A robust ensemble learning method that operates by constructing multiple decision trees during training and outputting the class that is the mode of the classes of the individual trees.
- **Performance:**
  - **Accuracy:** 87%
  - **Precision, Recall, F1-Score:** Precision: 0.86, Recall: 0.87, F1-Score: 0.85 
### Support Vector Machine (SVM)

- **Description:** A supervised machine learning algorithm that can be used for both classification or regression challenges. It performs classification by finding the hyperplane that best divides a dataset into classes.
- **Performance:**
  - **Accuracy:** 91%
  - **Precision, Recall, F1-Score:** Precision: 0.93, Recall: 0.91, F1-Score: 0.91

### Convolutional Neural Network (CNN)

- **Description:** A type of deep learning model, especially effective in capturing spatial dependencies in data. The CNN used here consists of multiple convolutional layers followed by fully connected layers.
- **Performance:**
  - **Accuracy:** 88%
  - **Precision, Recall, F1-Score:** Precision: 0.88, Recall: 0.88, F1-Score: 0.88 

## Evaluation

Each model was evaluated on the test set using the following metrics:
- **Accuracy:** The overall accuracy of the model in predicting the correct cancer type.
- **Precision, Recall, F1-Score:** Detailed in the classification report.
- **Confusion Matrix:** A matrix showing the true vs. predicted classifications.

## Visualizations

All visualizations are saved in the `visualizations` folder and include:
1. **Confusion Matrix:** Shows the performance of the classification models in predicting different classes.
2. **PCA Visualization:** A 2D representation of the test data using Principal Component Analysis (PCA).
3. **Training Loss Curves:** Plots of the training and validation loss over epochs for the CNN model.

## Future Work

1. **Model Tuning:** Further tuning of hyperparameters to improve performance, especially for minority classes.
2. **Advanced Architectures:** Experimentation with more advanced neural network architectures like ResNet or DenseNet.
3. **Class Imbalance Handling:** Implement techniques like SMOTE or a weighted loss function to address class imbalance.
4. **Integration of Clinical Data:** Combining gene expression data with other clinical data for more comprehensive models.

## Requirements

- **Python 3.x**
- **PyTorch**
- **scikit-learn**
- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**

## How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yonas650/AI-Powered-Gene-Sequencing.git
   cd AI-Powered-Gene-Sequencing
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Preprocessing:**
   ```bash
   python preprocess_data.py
   ```

4. **Train and Evaluate Models:**
   - **Random Forest:** `python random_forest.py`
   - **SVM:** `python svm.py`
   - **CNN:** `python neural_network.py`

5. **Review Visualizations:**
   - All generated visualizations will be saved in the `visualizations` folder.
