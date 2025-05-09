# Mushroom Classification Project

This repository contains a machine learning project for classifying mushrooms as edible or poisonous using the Mushroom dataset. The project implements data preprocessing, exploratory data analysis (EDA), and modeling with k-Nearest Neighbors (kNN) and Decision Tree classifiers.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Project Overview
The goal is to predict whether a mushroom is edible or poisonous based on 22 categorical features (e.g., cap-shape, odor, gill-color). Key components include:
- **Data Cleaning**: Handling missing values in `stalk-root` and encoding categorical variables.
- **EDA**: Visualizing feature distributions and correlations with edibility using count plots and Chi-Square tests.
- **Feature Engineering**: Dimensionality reduction via PCA and feature selection.
- **Modeling**: Training kNN and Decision Tree models with hyperparameter tuning via GridSearchCV.
- **Pipeline**: Reproducible preprocessing and modeling pipeline with custom transformers.

## Dataset
The [Mushroom Dataset](https://archive.ics.uci.edu/dataset/73/mushroom) (`agaricus-lepiota.data`) contains 8,124 instances:
- **Target**: Poisonous (p) or edible (e).
- **Features**: 22 categorical attributes describing mushroom characteristics.
- **Class Distribution**: Balanced (51.8% edible, 48.2% poisonous).
- **Missing Values**: Only in `stalk-root` (2,480 instances).

The dataset is located in the `mushroom/` directory.

## Installations
To run this project, you need Python 3.11+. Clone the repository and install the dependencies:

```bash
git clone https://github.com/Mahmoud-ABK/mushroom-edibility-prediction.git
cd mushroom-edibility-prediction
pip install -r requirements.txt
```

### Requirements
- numpy==1.26.4
- pandas==2.2.3
- matplotlib==3.10.0
- seaborn==0.13.2
- scipy==1.15.1
- scikit-learn==1.6.1
- contourpy==1.3.1
- cycler==0.11.0
- fonttools==4.55.3
- kiwisolver==1.4.8
- pillow==11.1.0
- pyparsing==3.2.0
- python-dateutil==2.9.0post0
- pytz==2024.1
- threadpoolctl==3.5.0

The full list is in `requirements.txt`.

## Project Structure
```plaintext
mushroom-edibility-prediction/
├── environment.yaml                  # Conda environment configuration
├── Mini project Mushroom.ipynb        # Main Jupyter notebook
├── mushroom/
│   ├── agaricus-lepiota.data         # Mushroom dataset
│   ├── agaricus-lepiota.names        # Dataset description
│   ├── expanded.Z                    # Compressed dataset
│   ├── full_dataset.csv              # Processed dataset
│   ├── Index                         # Dataset index
│   └── README                        # Dataset README
├── plots/
│   ├── all_stacked_bar_plots.png     # Stacked bar plots
│   ├── decision_tree_visualization.png # Decision tree visualization
│   ├── distributions.png             # Feature distribution plots
├── __pycache__/
│   ├── utils.cpython-311.pyc         # Compiled utils module (Python 3.11)
│   ├── utils.cpython-312.pyc         # Compiled utils module (Python 3.12)
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
└── utils.py                          # Custom utility functions
```

## Acknowledgements
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/73/mushroom) for providing the dataset.
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, SciPy, Scikit-learn.