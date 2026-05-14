# Heart Disease Prediction using K-Nearest Neighbour in C

## Overview

This project is a machine learning healthcare application built in C to predict the presence of heart disease using the K-Nearest Neighbour (KNN) algorithm.

The model was implemented from scratch without using external machine learning libraries. It includes CSV data loading, data preprocessing, Min-Max normalisation, train/test splitting, K-value optimisation, prediction, and evaluation using accuracy, precision, recall and F1-score.

The project uses the UCI Cleveland Heart Disease dataset.

## Key Features

- K-Nearest Neighbour classifier built from scratch in C
- Euclidean distance calculation
- Majority voting classification
- CSV dataset loading
- Fisher-Yates data shuffling
- Min-Max feature normalisation
- 80/20 train/test split
- Search for optimal odd K values from 1 to 15
- Confusion matrix evaluation
- Accuracy, precision, recall and F1-score calculation
- Interactive patient input prediction mode

## Technologies Used

- C programming language
- Visual Studio / Visual Studio Code
- UCI Cleveland Heart Disease dataset
- Command-line interface

## Project Structure

```text
src/
├── main.c
├── knn.c
├── knn.h
└── data_loader.c

data/
└── data_set.csv

docs/
└── project-report.pdf
