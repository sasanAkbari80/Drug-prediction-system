# Drug Prediction using Decision Tree Classifier

This project is a machine learning application that predicts the appropriate drug type for a patient based on medical features such as age, sex, blood pressure, cholesterol, and sodium-to-potassium ratio.

## Project Overview

This project uses a Decision Tree model to classify drug types based on patient data.  
It includes data preprocessing, model training, and evaluation in a modular structure.

## Dataset

The dataset contains the following features:
- Age
- Sex
- BP
- Cholesterol
- Na_to_K
- Drug (target)

## Project Structure

drug-prediction/
│
├── data/
│   └── drug200.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── main.py
│
├── requirements.txt
└── README.md

## Installation

pip install -r requirements.txt

## How to Run

cd src  
python main.py

## Model Details

- Algorithm: Decision Tree Classifier  
- Criterion: Entropy  
- Max Depth: 4  
- Train/Test Split: 70/30

## Output

- Accuracy score  
- Sample predictions
