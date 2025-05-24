# Bitcoin Price Prediction Project

## Project Structure
```
Project_Root/
├── data/
│   ├── BTC_train.csv           # Training dataset for Bitcoin price prediction
│   └── BTC_test.csv            # Test dataset for Bitcoin price prediction
│
├── gp_classifier/              # Genetic Programming classifier (Java)
│   ├── src/
│   │   ├── Node.java           # Interface for GP nodes
│   │   ├── FeatureNode.java    # Terminal node representing features
│   │   ├── FunctionNodes/      # Function nodes (Add, Subtract, Multiply, Divide)
│   │   ├── Individual.java     # GP individual (tree structure)
│   │   ├── Population.java     # Population management for GP
│   │   ├── DataParser.java     # Parses CSV data
│   │   └── GPClassifier.java   # Main class for GP classifier
│   ├── bin/                    # Compiled Java classes
│   ├── README.md               # Instructions specific to GP classifier
│   └── run_gp.sh               # Script to run GP: java -jar GP.jar <seed> <train> <test>
│
├── MLP/                        # Multi-Layer Perceptron classifier (Python)
│   ├── StockPredictor.py       # MLP implementation using TensorFlow and numpy
│   ├── requirements.txt        # Python dependencies (e.g., numpy, scikit-learn)
│   └── trained_btc_model.keras # Previously trained model
│    
├── decision_tree/
│   ├── dt_model.arff           # ARFF-formatted training/test data (converted from CSV)
│   ├── J48_DecisionTree.java   # Java code to run J48 using Weka
│   ├── weka.jar                # Weka library (if not globally installed)
│   └── run_dt.sh               # Script to execute: `java -cp weka.jar J48_DecisionTree`
├── report/                     # Project report
│   ├── report.tex              # LaTeX source for the group report
│   └── results_table.tex       # LaTeX table with results and statistical tests
├── results/                    # Output predictions
│   ├── gp_predictions.csv      # Predictions from GP classifier
│   ├── mlp_predictions.csv     # Predictions from MLP classifier
│   └── dt_predictions.csv      # Predictions from Decision Tree classifier
├── scripts/                    # Utility scripts for evaluation
│   ├── evaluate.py             # Python script to calculate accuracy and F1-score
│   └── wilcoxon_test.R         # R script for Wilcoxon signed-rank test (GP vs. MLP)
└── README.md                   # This file
```
# Genetic Programming Algorithm
### How to Run

# Multi Layer Perceptron Algorithm
### How to Run
* Enter the MLP directory using `cd MLP`
* Install dependencies by running `pip install -r requirements.txt`
* To run with previously trained model, simply run `python StockPredictor.py`
* To first retrain a new model and then run, first delete `trained_bt_model.keras` then run `python StockPredictor.py`

# Decision Tree Algorithm
### How to Run