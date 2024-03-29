# Breast Cancer Prediction Project

This repository contains a breast cancer prediction project implemented using machine learning techniques and neural networks. The project aims to predict whether a given tumor is malignant or benign based on various features extracted from diagnostic images.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) dataset, available on [Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data). It comprises features computed from digitized images of breast mass, which are then used to predict the diagnosis of the tumor.

## Technologies Used
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow
- Keras
- Matplotlib

## Project Overview
1. **Data Preprocessing:** The dataset is loaded using Scikit-learn's `load_breast_cancer` function. It is then split into training and testing sets, and standard scaling is applied to normalize the feature values.
2. **Model Development:** Two neural network models are built using Keras Sequential API. The first model uses mean squared error as the loss function, while the second model uses binary cross-entropy. Both models consist of input, hidden, and output layers with different activation functions.
3. **Model Training:** The models are trained on the training dataset using the Adam optimizer. Training is performed for multiple epochs, and the models' performance is evaluated using accuracy metrics.
4. **Model Evaluation:** The trained models are evaluated on the test dataset to assess their performance. Test loss and accuracy metrics are computed to measure the models' effectiveness in predicting breast cancer diagnoses.

## Results
- The model using binary cross-entropy achieved a test accuracy of approximately 97.37%.
- The model's test loss was approximately 0.1037.

## File Structure
- `breast_cancer_prediction.ipynb`: Jupyter Notebook containing the Python code for data preprocessing, model development, training, and evaluation.
- `README.md`: Markdown file containing project overview, dataset information, technologies used, project structure, and results.

## Usage
1. Clone the repository: `git clone https://github.com/your-username/breast-cancer-prediction.git`
2. Navigate to the project directory: `cd breast-cancer-prediction`
3. Open and run the `breast_cancer_prediction.ipynb` notebook using Jupyter or any compatible environment.

## Contributors
- Arya Sahu
