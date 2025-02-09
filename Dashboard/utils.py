import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def load_abt_files():
    """Load all ABT_score files and combine them with appropriate target labels"""
    abt_files = list(Path('../data').glob('ABT_[Ss]core_*.csv'))
    
    if not abt_files:
        print("No ABT_score_*.csv files found in current directory!")
        print("\nCurrent directory contents:")
        print([f.name for f in Path('../data').glob('*')])
        print("\nPlease ensure your ABT_score_*.csv files are in the data directory.")
        return None
    
    dfs = []
    for file_path in abt_files:
        product = file_path.stem.split('_')[-1]
        try:
            print(f"\nLoading {product} data...")
            df = pd.read_csv(file_path, sep=';')
            print(f"Successfully loaded {len(df)} rows for {product}")
            df['product_type'] = product
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file_path.name}: {str(e)}")
    
    return pd.concat(dfs, ignore_index=True)

def preprocess_data(df):
    """Clean and preprocess the combined dataset"""
    df = df.copy()
    
    # Convert date columns to datetime
    date_columns = [col for col in df.columns if 'Date' in col or 'date' in col or 
                   col.startswith(('mFirst_', 'mLast_'))]
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Fill numeric NaNs with 0
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Convert binary columns to int
    binary_columns = [col for col in df.columns if col.startswith(('Have_', 'Had_', 'Optout_'))]
    for col in binary_columns:
        df[col] = df[col].fillna(0).astype(int)
    
    return df

def analyze_product_sequence(df):
    """Analyze the sequence of products purchased by customers"""
    if len(df) == 0:
        raise ValueError("Empty dataframe provided!")
        
    product_cols = [col for col in df.columns if col.startswith('mFirst_')]
    if not product_cols:
        raise ValueError("No product columns found!")
    
    timeline_data = []
    customer_journeys = {}
    
    for customer_id in df['sCustomerNaturalKey'].unique():
        customer_data = df[df['sCustomerNaturalKey'] == customer_id]
        
        # Get product acquisition dates
        products = []
        for col in product_cols:
            product = col.replace('mFirst_', '')
            date = customer_data[col].iloc[0]
            if pd.notna(date):
                products.append({
                    'sCustomerNaturalKey': customer_id,
                    'product': product,
                    'acquisition_date': date
                })
        
        # Sort products by date
        products = sorted(products, key=lambda x: x['acquisition_date'])
        timeline_data.extend(products)
        
        # Create journey sequence
        if products:
            journey = ' → '.join([p['product'] for p in products])
            customer_journeys[customer_id] = {
                'sequence': journey,
                'length': len(products),
                'duration_days': (products[-1]['acquisition_date'] - products[0]['acquisition_date']).days,
                'first_product': products[0]['product'],
                'last_product': products[-1]['product']
            }
    
    return pd.DataFrame(timeline_data), pd.DataFrame.from_dict(customer_journeys, orient='index')

def analyze_lifecycle_stages(journey_df, combined_df):
    """Analyze customer lifecycle stages and transitions"""
    lifecycle_data = pd.DataFrame()
    
    # Define lifecycle stages based on journey length
    lifecycle_data['stage'] = pd.cut(
        journey_df['length'], 
        bins=[0, 1, 3, 5, float('inf')],
        labels=['New', 'Growing', 'Established', 'Mature'],
        include_lowest=True
    )
    
    # Calculate adoption rate
    lifecycle_data['adoption_rate'] = journey_df['length'] / journey_df['duration_days']
    lifecycle_data['adoption_rate'] = lifecycle_data['adoption_rate'].fillna(0).replace([np.inf, -np.inf], 0)
    
    # Drop any rows where either stage or adoption_rate is null
    lifecycle_data = lifecycle_data.dropna(subset=['stage', 'adoption_rate'])
    
    # Ensure adoption_rate is not infinite
    lifecycle_data = lifecycle_data[lifecycle_data['adoption_rate'] < 1]  # Filter out unrealistic rates
    
    return lifecycle_data

def analyze_journey_patterns(journey_df):
    """Analyze patterns in customer journeys"""
    return {
        'journey_stats': {
            'total_customers': len(journey_df),
            'avg_products': journey_df['length'].mean(),
            'avg_duration': journey_df['duration_days'].mean(),
            'common_first': journey_df['first_product'].value_counts().head(),
            'common_last': journey_df['last_product'].value_counts().head()
        },
        'journey_segments': {
            'single_product': (journey_df['length'] == 1).mean(),
            'short_journey': ((journey_df['length'] > 1) & (journey_df['length'] <= 3)).mean(),
            'long_journey': (journey_df['length'] > 3).mean()
        }
    }

def analyze_churn_risk(journey_df, combined_df, timeline_df):
    """Analyze potential churn indicators in customer journeys"""
    risk_factors = pd.DataFrame()
    
    # Time since last product
    current_date = combined_df['mFirst_BankBolan'].max()  # Use as reference date
    # Using the last product's acquisition date
    for idx in journey_df.index:
        last_product = journey_df.loc[idx, 'last_product']
        last_date = timeline_df[timeline_df['product'] == last_product]['acquisition_date'].max()
        risk_factors.loc[idx, 'days_since_last_product'] = (current_date - last_date).days
    
    # Product discontinuation
    had_cols = [col for col in combined_df.columns if col.startswith('Had_')]
    have_cols = [col.replace('Had_', 'Have_') for col in had_cols]
    
    risk_factors['discontinued_products'] = 0
    for had, have in zip(had_cols, have_cols):
        risk_factors['discontinued_products'] += (combined_df[had] > combined_df[have]).astype(int)
    
    return risk_factors