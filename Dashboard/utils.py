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
    
    combined_df = pd.concat(dfs, ignore_index=True)
    try:
        validate_data(combined_df)
    except ValueError as e:
        print(f"Data validation failed: {str(e)}")
        return None
        
    return combined_df

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

def analyze_product_sequence(df):
    """Analyze the sequence of products purchased by customers"""
    if len(df) == 0:
        raise ValueError("Empty dataframe provided!")
        
    product_cols = [col for col in df.columns if col.startswith('mFirst_')]
    if not product_cols:
        raise ValueError("No product columns found!")
    
    timeline_data = []
    customer_journeys = {}
    
    # Convert date columns and filter invalid dates
    for col in product_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove rows with all missing product dates
    df = df.dropna(subset=product_cols, how='all')
    
    for customer_id in df['sCustomerNaturalKey'].unique():
        customer_data = df[df['sCustomerNaturalKey'] == customer_id]
        
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
        
        if len(products) > 1:  # Only process customers with multiple products
            products = sorted(products, key=lambda x: x['acquisition_date'])
            timeline_data.extend(products)
            
            journey = ' → '.join([p['product'] for p in products])
            duration = (products[-1]['acquisition_date'] - products[0]['acquisition_date']).days
            
            # Ensure duration is at least 1 day
            duration = max(1, duration)
            
            customer_journeys[customer_id] = {
                'sequence': journey,
                'length': len(products),
                'duration_days': duration,
                'first_product': products[0]['product'],
                'last_product': products[-1]['product']
            }
    
    timeline_df = pd.DataFrame(timeline_data)
    journey_df = pd.DataFrame.from_dict(customer_journeys, orient='index')
    
    return timeline_df, journey_df

def analyze_lifecycle_stages(journey_df, combined_df):
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
    current_date = combined_df['mFirst_BankBolan'].max()  # Use as reference date
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
    Create an improved product adoption timeline visualization
    """
    if timeline_df.empty:
        return go.Figure()
        
    # Ensure dates are datetime objects and filter out invalid dates
    timeline_df = timeline_df.dropna(subset=['acquisition_date'])
    timeline_df['acquisition_date'] = pd.to_datetime(timeline_df['acquisition_date'])
    
    # Create numeric y-axis positions for products
    unique_products = timeline_df['product'].unique()
    product_mapping = {product: idx for idx, product in enumerate(unique_products)}
    
    # Create color mapping
    color_sequence = px.colors.qualitative.Set3
    color_map = {product: color_sequence[i % len(color_sequence)] 
                 for i, product in enumerate(unique_products)}
    
    # Create the figure
    fig = go.Figure()
    
    for product in unique_products:
        product_data = timeline_df[timeline_df['product'] == product]
        
        if not product_data.empty:
            # Generate jittered y-positions using numeric values
            base_y = product_mapping[product]
            y_jitter = np.random.uniform(-0.3, 0.3, len(product_data))
            y_positions = base_y + y_jitter
            
            fig.add_trace(go.Scatter(
                x=product_data['acquisition_date'],
                y=y_positions,
                name=product,
                mode='markers',
                marker=dict(
                    color=color_map[product],
                    size=8,
                    line=dict(width=1, color='black')
                ),
                hovertemplate="Product: %{text}<br>Date: %{x|%Y-%m-%d}<extra></extra>",
                text=[product]*len(product_data)
            ))
    
    # Configure y-axis to show product names
    fig.update_layout(
        title="Product Adoption Timeline",
        xaxis_title="Acquisition Date",
        yaxis_title="Product",
        height=600,
        showlegend=True,
        hovermode='closest',
        yaxis=dict(
            tickmode='array',
            tickvals=list(product_mapping.values()),
            ticktext=list(product_mapping.keys()),
            range=[-0.5, len(unique_products)-0.5]
        )
    )
    
    return fig

def plot_customer_journey_sankey(journey_df, max_paths=20, min_customers=50):
    """
    Create an enhanced Sankey diagram with more steps and color coding
    """
    if journey_df.empty:
        return go.Figure()
        
    # Get sequences with their counts
    sequence_counts = journey_df['sequence'].value_counts()
    sequence_counts = sequence_counts[sequence_counts >= min_customers].head(max_paths)
    
    if len(sequence_counts) == 0:
        sequence_counts = journey_df['sequence'].value_counts().head(max_paths)
    
    # Create nodes and links
    nodes = set()
    links = []
    link_values = []
    link_colors = []
    
    # Create color mapping based on first product
    first_products = set(seq.split(' → ')[0] for seq in sequence_counts.index)
    color_sequence = px.colors.qualitative.Set3
    first_product_colors = {prod: color_sequence[i % len(color_sequence)] 
                           for i, prod in enumerate(first_products)}
    
    # Process each sequence
    for sequence, count in sequence_counts.items():
        products = sequence.split(' → ')
        nodes.update(products)
        
        # Determine the color based on first product
        sequence_color = first_product_colors[products[0]]
        
        # Add links between consecutive products
        for i in range(len(products) - 1):
            links.append((products[i], products[i + 1]))
            link_values.append(count)
            link_colors.append(sequence_color)
    
    # Convert nodes to list and create indices
    nodes = list(nodes)
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Create node colors (neutral color for all nodes)
    node_colors = ['rgba(200,200,200,0.5)' for _ in nodes]
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate='Node: %{label}<br>Total Flow: %{value}<extra></extra>'
        ),
        link=dict(
            source=[node_indices[link[0]] for link in links],
            target=[node_indices[link[1]] for link in links],
            value=link_values,
            color=link_colors,
            hovertemplate='From: %{source.label}<br>To: %{target.label}<br>Flow: %{value}<extra></extra>'
        )
    )])
    
    fig.update_layout(
        title="Customer Journey Paths Analysis",
        font_size=12,
        height=800,
        showlegend=True
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