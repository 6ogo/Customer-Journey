# Customer Journey Analysis Project

## Overview
This project analyzes customer journeys and product adoption patterns using data from multiple financial and insurance products. It includes visualization of customer behavior, demographic analysis, and optional predictive modeling capabilities.

## Data Structure
The project expects CSV files with the following naming convention:
- `ABT_score_(PRODUCT).csv
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

### Python Dependencies
```bash
pip install pandas numpy torch matplotlib seaborn plotly jax scikit-learn
```

### System Requirements
- Python 3.8+
- Jupyter Notebook/JupyterLab
- GPU (optional, for accelerated computing with PyTorch/JAX)

## Project Structure
```
customer-journey-analysis/
├── data/
│   ├── ABT_score_(PRODUCT).csv
│   └── ...
├── notebooks/
│   └── customer_journey_analysis.ipynb
├── README.md
└── requirements.txt
```

## Features

### 1. Data Processing
- Automated loading of all ABT_score files
- Data cleaning and preprocessing
- Date normalization
- Binary feature encoding

### 2. Analysis Capabilities
- Customer journey mapping
- Product sequence analysis
- Demographic distribution analysis
- Product adoption timeline visualization
- Product combination analysis

### 3. Visualizations
- Interactive Sankey diagrams of customer journeys
- Demographic distribution plots
- Product adoption timelines
- Correlation heatmaps of product combinations

### 4. Optional Machine Learning
- PyTorch-based predictive modeling
- Feature preparation and scaling
- Neural network architecture for product adoption prediction

## Usage

1. Clone the repository:
```bash
git clone https://github.com/6ogo/Customer-Journey
cd Customer-Journey
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Place your ABT_score CSV files in the data directory

4. Launch Jupyter Notebook:
```bash
jupyter notebook
```

5. Open `notebooks/customer_journey_analysis.ipynb` and run cells sequentially

## Analysis Outputs

### Customer Journey Analysis
- Visualization of common product adoption sequences
- Identification of typical first products
- Time between product acquisitions

### Demographic Analysis
- Age distribution by product
- Gender distribution by product
- Demographic segmentation insights

### Product Analysis
- Product combination patterns
- Correlation between different products
- Adoption timeline patterns

## Contributing
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Future Enhancements
- Additional cohort analysis capabilities
- Customer lifetime value calculations
- Churn prediction modeling
- More detailed demographic segmentation
- Streamlit Dashboard