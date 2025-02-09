# Customer Journey Analysis Project

## Overview
This project analyzes customer journeys and product adoption patterns using data from multiple financial and insurance products. It provides both a detailed Jupyter notebook analysis and an interactive Streamlit dashboard for exploring customer behavior, demographic analysis, and predictive modeling capabilities.

## Data Structure
The project expects CSV files with the following naming convention:
- `ABT_score_(PRODUCT).csv`
- Additional ABT_score files can be added following the same pattern

### Key Data Fields
- `myTarget`: Binary indicator (1 for customers who bought the product, 0 for random sample)
- `sCustomerNaturalKey`: Unique customer identifier
- Various product-related fields with prefixes:
  - `mFirst_`: First purchase date of product
  - `mLastStart_`: Latest start date of product
  - `Have_`: Current ownership status
  - `Had_`: Historical ownership status
  - `nbr_active_agr_`: Number of active agreements

## Requirements

The project has two main components with separate requirements:

### Jupyter Notebook Analysis
Located in `/notebook`, with requirements:
```bash
cd notebook
pip install -r requirements.txt
```
This includes:
- pandas
- numpy
- torch
- matplotlib
- seaborn
- jax
- scikit-learn
- plotly

### Streamlit Dashboard
Located in `/Dashboard`, with requirements:
```bash
cd Dashboard
pip install -r requirements.txt
```
This includes:
- streamlit
- pandas
- numpy
- plotly
- seaborn
- matplotlib

### System Requirements
- Python 3.8+
- Jupyter Notebook/JupyterLab (for analysis notebook)
- GPU (optional, for accelerated computing with PyTorch/JAX)

## Project Structure
```
customer-journey-analysis/
├── data/
│   ├── ABT_score_(PRODUCT).csv
│   └── ...
├── notebook/
│   ├── customer_journey_analysis.ipynb
│   └── requirements.txt
├── Dashboard/
│   ├── customerJourney.py
│   ├── utils.py
│   └── requirements.txt
└── README.md
```

## Features

### 1. Jupyter Notebook Analysis
- Automated loading of all ABT_score files
- Data cleaning and preprocessing
- Customer journey mapping
- Product sequence analysis
- Demographic distribution analysis
- Product adoption timeline visualization
- Interactive Sankey diagrams
- Optional machine learning with PyTorch

### 2. Streamlit Dashboard
- Interactive visualization of customer journeys
- Product analysis and correlations
- CRM recommendations
- Demographic insights
- Churn risk analysis
- Product transition patterns

## Usage

1. Clone the repository:
```bash
git clone https://github.com/6ogo/Customer-Journey
cd Customer-Journey
```

2. Place your ABT_score CSV files in the data directory

3. For Jupyter Notebook Analysis:
```bash
cd notebook
pip install -r requirements.txt
jupyter notebook
```
Then open `customer_journey_analysis.ipynb`

4. For Streamlit Dashboard:
```bash
cd Dashboard
pip install -r requirements.txt
streamlit run customerJourney.py
```

## Analysis Outputs

### Customer Journey Analysis
- Visualization of common product adoption sequences
- Identification of typical first products
- Time between product acquisitions
- Demographic segmentation insights
- Product combination patterns
- Adoption timeline patterns

### Dashboard Features
- Overview & Journey Analysis
- Product Analysis
- CRM & Recommendations
- Churn Risk Assessment
- Interactive Visualizations

## Contributing
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request