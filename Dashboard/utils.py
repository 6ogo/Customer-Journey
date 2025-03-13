import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

def validate_data(df):
    """Validate input data quality and required columns"""
    required_cols = ['sCustomerNaturalKey', 'Age', 'Woman', 'Apartment']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate date columns
    date_cols = [col for col in df.columns if col.startswith('mFirst_')]
    if not date_cols:
        raise ValueError("No product date columns found")
        
    return True

def load_csv_file(file_path):
    """Load the CSV file containing customer data"""
    try:
        df = pd.read_csv(file_path)
        validate_data(df)
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

def preprocess_data(df):
    """Clean and preprocess the combined dataset"""
    if df is None or df.empty:
        raise ValueError("Empty or None dataframe provided")
        
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

# In utils.py - modify the analyze_product_sequence function
def analyze_product_sequence(df):
    """Analyze expanded sequence of products including multiple touches"""
    if len(df) == 0:
        raise ValueError("Empty dataframe provided!")
        
    # Add error handling for missing columns
    product_cols = [col for col in df.columns if col.startswith('mFirst_')]
    if not product_cols:
        raise ValueError("No product columns (mFirst_*) found in the data")
    
    # Identify different sequence patterns (first, second, third, etc.)
    sequence_patterns = ['mFirst_']  # Start with just first purchases
    
    # Create a comprehensive timeline for each customer
    timeline_data = []
    customer_journeys = {}
    
    for customer_id in df['sCustomerNaturalKey'].unique():
        customer_data = df[df['sCustomerNaturalKey'] == customer_id].iloc[0]
        
        # Collect all product touches for this customer
        all_touches = []
        
        for pattern in sequence_patterns:
            # Get columns that match this pattern
            pattern_cols = [col for col in df.columns if col.startswith(pattern)]
            
            for col in pattern_cols:
                product = col.replace(pattern, '')
                date = customer_data[col]
                
                if pd.notna(date):
                    # Add sequence number information
                    seq_num = sequence_patterns.index(pattern) + 1
                    all_touches.append({
                        'sCustomerNaturalKey': customer_id,
                        'product': product,
                        'acquisition_date': date,
                        'sequence_number': seq_num,
                        'product_with_seq': f"{product} ({seq_num})"
                    })
        
        # Sort by date and create journey
        if all_touches:
            all_touches = sorted(all_touches, key=lambda x: x['acquisition_date'])
            timeline_data.extend(all_touches)
            
            # Create two journey representations
            simple_journey = ' → '.join([t['product'] for t in all_touches])
            detailed_journey = ' → '.join([t['product_with_seq'] for t in all_touches])
            
            # Handle edge case where dates might be invalid
            try:
                duration = (all_touches[-1]['acquisition_date'] - all_touches[0]['acquisition_date']).days
                duration = max(1, duration)
            except:
                duration = 1
            
            customer_journeys[customer_id] = {
                'sequence': simple_journey,  # Standard sequence without seq numbers
                'detailed_sequence': detailed_journey,  # With sequence numbers
                'length': len(all_touches),
                'duration_days': duration,
                'first_product': all_touches[0]['product'],
                'last_product': all_touches[-1]['product']
            }
    
    # Create DataFrames with proper error handling
    if not timeline_data:
        timeline_df = pd.DataFrame(columns=['sCustomerNaturalKey', 'product', 'acquisition_date', 
                                           'sequence_number', 'product_with_seq'])
    else:
        timeline_df = pd.DataFrame(timeline_data)
    
    if not customer_journeys:
        journey_df = pd.DataFrame(columns=['sequence', 'detailed_sequence', 'length', 
                                          'duration_days', 'first_product', 'last_product'])
    else:
        journey_df = pd.DataFrame.from_dict(customer_journeys, orient='index')
    
    return timeline_df, journey_df

def analyze_lifecycle_stages(journey_df):
    """Analyze customer lifecycle stages and transitions"""
    if journey_df.empty:
        return pd.DataFrame()
        
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
    
    # Ensure adoption_rate is not infinite and is realistic
    lifecycle_data = lifecycle_data[lifecycle_data['adoption_rate'] < 1]
    
    return lifecycle_data

def analyze_journey_patterns(journey_df):
    """Analyze patterns in customer journeys"""
    if journey_df.empty:
        return {
            'journey_stats': {
                'total_customers': 0,
                'avg_products': 0,
                'avg_duration': 0,
                'common_first': pd.Series()
            },
            'journey_segments': {
                'single_product': 0,
                'short_journey': 0,
                'long_journey': 0
            }
        }
        
    return {
        'journey_stats': {
            'total_customers': len(journey_df),
            'avg_products': journey_df['length'].mean(),
            'avg_duration': journey_df['duration_days'].mean(),
            'common_first': journey_df['first_product'].value_counts().head()
        },
        'journey_segments': {
            'single_product': (journey_df['length'] == 1).mean(),
            'short_journey': ((journey_df['length'] > 1) & (journey_df['length'] <= 3)).mean(),
            'long_journey': (journey_df['length'] > 3).mean()
        }
    }

def analyze_churn_risk(journey_df, combined_df, timeline_df):
    """Analyze potential churn indicators in customer journeys"""
    if any(df.empty for df in [journey_df, combined_df, timeline_df]):
        return pd.DataFrame()
        
    risk_factors = pd.DataFrame()
    
    # Time since last product
    current_date = combined_df['mFirst_BankBolån'].max()  # Use as reference date
    if pd.isna(current_date):
        return pd.DataFrame()
        
    # Using the last product's acquisition date
    for idx in journey_df.index:
        last_product = journey_df.loc[idx, 'last_product']
        last_date = timeline_df[timeline_df['product'] == last_product]['acquisition_date'].max()
        if pd.notna(last_date):
            risk_factors.loc[idx, 'days_since_last_product'] = (current_date - last_date).days
    
    # Product discontinuation
    had_cols = [col for col in combined_df.columns if col.startswith('Had_')]
    have_cols = [col.replace('Had_', 'Have_') for col in had_cols]
    
    risk_factors['discontinued_products'] = 0
    for had, have in zip(had_cols, have_cols):
        risk_factors['discontinued_products'] += (combined_df[had] > combined_df[have]).astype(int)
    
    return risk_factors

def create_product_timeline(timeline_df):
    """
    Create a product adoption timeline visualization using a categorical y-axis.
    """
    if timeline_df.empty:
        return go.Figure()

    # Ensure dates are valid datetime objects
    timeline_df = timeline_df.dropna(subset=['acquisition_date'])
    timeline_df['acquisition_date'] = pd.to_datetime(timeline_df['acquisition_date'], errors='coerce')
    timeline_df = timeline_df.dropna(subset=['acquisition_date'])
    
    # Get a sorted list of unique products for the y-axis
    unique_products = sorted(timeline_df['product'].unique())
    
    # Create a consistent color mapping for each product
    color_sequence = px.colors.qualitative.Set3
    color_map = {product: color_sequence[i % len(color_sequence)] for i, product in enumerate(unique_products)}
    
    fig = go.Figure()

    # For each product, add a trace with y-axis as the product name (category)
    for product in unique_products:
        product_data = timeline_df[timeline_df['product'] == product]
        if product_data.empty:
            continue
        
        fig.add_trace(go.Scatter(
            x=product_data['acquisition_date'],
            y=[product] * len(product_data),  # Use the product name as the y value
            mode='markers',
            marker=dict(
                color=color_map[product],
                size=8,
                line=dict(width=1, color='black')
            ),
            name=product,
            hovertemplate="Product: %{y}<br>Date: %{x|%Y-%m-%d}<extra></extra>"
        ))
    
    fig.update_layout(
        title="Product Adoption Timeline",
        xaxis_title="Acquisition Date",
        yaxis_title="Product",
        height=600,
        showlegend=True,
        hovermode='closest',
        yaxis=dict(type='category')
    )
    
    return fig

def plot_lifecycle_analysis(lifecycle_data):
    """
    Create an improved lifecycle stage analysis visualization
    
    Parameters:
    -----------
    lifecycle_data : pd.DataFrame
        DataFrame containing lifecycle stages and adoption rates
    
    Returns:
    --------
    go.Figure or None
        Box plot visualization of adoption rates by lifecycle stage
    """
    if lifecycle_data.empty:
        return None
        
    fig = go.Figure()
    
    # Add box plot for adoption rates
    fig.add_trace(go.Box(
        y=lifecycle_data['adoption_rate'],
        x=lifecycle_data['stage'],
        name='Adoption Rate',
        marker_color='rgb(107,174,214)',
        boxpoints='all',
        jitter=0.3,
        pointpos=-1.8
    ))
    
    fig.update_layout(
        title='Adoption Rate by Lifecycle Stage',
        xaxis_title='Lifecycle Stage',
        yaxis_title='Adoption Rate',
        height=500,
        showlegend=False
    )
    
    return fig