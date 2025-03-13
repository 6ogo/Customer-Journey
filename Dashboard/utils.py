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

def plot_customer_journey_sankey(journey_df, max_paths=20, min_customers=50, color_map=None):
    """
    Create an enhanced Sankey diagram for customer journeys using a fixed color per product.
    Improved for better visualization and centering on the screen.
    
    Parameters:
    -----------
    journey_df : pd.DataFrame
        DataFrame with customer journey data. Must include a 'sequence' column 
        (each sequence is a string with products separated by ' → ').
    max_paths : int, optional
        Maximum number of journey paths to include (default is 20).
    min_customers : int, optional
        Minimum number of customers per journey path required to include that path (default is 50).
    color_map : dict, optional
        Dictionary mapping each product name to a color string. If not provided, it will be
        computed based on the products found in the journey sequences.
        
    Returns:
    --------
    go.Figure
        A Plotly Figure object containing the Sankey diagram.
    """
    if journey_df.empty:
        return go.Figure()

    # Get sequences with their counts
    sequence_counts = journey_df['sequence'].value_counts()
    sequence_counts = sequence_counts[sequence_counts >= min_customers].head(max_paths)
    if len(sequence_counts) == 0:
        sequence_counts = journey_df['sequence'].value_counts().head(max_paths)

    # If no color_map is provided, compute one using the products present in these sequences.
    all_products = set()
    for sequence in sequence_counts.index:
        all_products.update(sequence.split(" → "))
    all_products = sorted(all_products)
    
    if color_map is None:
        # Use brighter, more distinct colors
        color_palette = [
            '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
            '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
            '#AECDE8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5'
        ]
        color_map = {prod: color_palette[i % len(color_palette)] for i, prod in enumerate(all_products)}

    # Create nodes and links
    nodes = []
    links = []
    link_values = []
    link_colors = []
    link_labels = []
    node_map = {}  # mapping product name to node index

    # Process each journey sequence
    for sequence, count in sequence_counts.items():
        products = sequence.split(" → ")
        for i, prod in enumerate(products):
            if prod not in node_map:
                node_map[prod] = len(nodes)
                nodes.append(prod)
            # Create a link from the current product to the next one (if any)
            if i < len(products) - 1:
                source_idx = node_map[prod]
                target_idx = node_map.get(products[i + 1])
                
                # If target is not yet in node_map, add it
                if target_idx is None:
                    target_idx = len(nodes)
                    node_map[products[i + 1]] = target_idx
                    nodes.append(products[i + 1])
                
                links.append((source_idx, target_idx))
                link_values.append(count)
                # Use the color for the source product for the link
                link_colors.append(color_map.get(prod, "gray"))
                link_labels.append(f"{prod} → {products[i + 1]}: {count} customers")

    # Build a list of node colors based on the provided color_map
    node_colors = [color_map.get(prod, "gray") for prod in nodes]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
            hovertemplate='<b>%{label}</b><br>Total Customers: %{value}<extra></extra>'
        ),
        link=dict(
            source=[link[0] for link in links],
            target=[link[1] for link in links],
            value=link_values,
            color=[f"rgba{tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" 
                  for c in link_colors],
            customdata=link_labels,
            hovertemplate='<b>%{customdata}</b><extra></extra>'
        ),
        arrangement='snap'
    )])
    
    fig.update_layout(
        title={
            'text': "Enhanced Customer Journey Sankey Diagram",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        font=dict(size=16, color="white"),
        width=1000,
        height=800,
        autosize=True,
        margin=dict(l=25, r=25, t=50, b=25),
        paper_bgcolor="#111111",
        plot_bgcolor="#111111",
    )
    
    return fig

def plot_animated_journey_sankey(journey_df, max_paths=20, min_customers=50):
    """
    Create an animated Sankey diagram that shows the customer journey flow
    step by step for better understanding.
    
    Parameters:
    -----------
    journey_df : pd.DataFrame
        DataFrame with customer journey data.
    max_paths : int, optional
        Maximum number of journey paths to include.
    min_customers : int, optional
        Minimum number of customers per journey path.
        
    Returns:
    --------
    go.Figure
        An animated Plotly Figure object.
    """
    if journey_df.empty:
        return go.Figure()
    
    # Get journey sequences sorted by count
    sequence_counts = journey_df['sequence'].value_counts()
    sequence_counts = sequence_counts[sequence_counts >= min_customers].head(max_paths)
    
    if len(sequence_counts) == 0:
        sequence_counts = journey_df['sequence'].value_counts().head(max_paths)
    
    # Get unique products across all journeys
    all_products = set()
    max_journey_length = 0
    
    for sequence in sequence_counts.index:
        products = sequence.split(" → ")
        all_products.update(products)
        max_journey_length = max(max_journey_length, len(products))
    
    all_products = sorted(all_products)
    
    # Create color map
    color_palette = [
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
        '#AECDE8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5'
    ]
    color_map = {prod: color_palette[i % len(color_palette)] for i, prod in enumerate(all_products)}
    
    # Create frames for animation
    frames = []
    
    # Process each stage of the journey (from 1 to max_journey_length)
    for stage in range(1, max_journey_length):
        nodes = []
        node_map = {}
        links = []
        link_values = []
        link_colors = []
        link_labels = []
        
        # Process each journey sequence up to the current stage
        for sequence, count in sequence_counts.items():
            products = sequence.split(" → ")
            if len(products) <= stage:
                continue
                
            # Add nodes for this sequence
            for i, prod in enumerate(products[:stage+1]):
                if prod not in node_map:
                    node_map[prod] = len(nodes)
                    nodes.append(prod)
                
                # Add links between consecutive products
                if i < stage:
                    source_idx = node_map[prod]
                    target_idx = node_map[products[i+1]]
                    links.append((source_idx, target_idx))
                    link_values.append(count)
                    link_colors.append(color_map.get(prod, "gray"))
                    link_labels.append(f"{prod} → {products[i+1]}: {count} customers")
        
        # Skip if no links for this stage
        if not links:
            continue
            
        # Build node colors
        node_colors = [color_map.get(prod, "gray") for prod in nodes]
        
        # Create Sankey for this stage
        frame = go.Frame(
            data=[go.Sankey(
                node=dict(
                    pad=20,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=nodes,
                    color=node_colors,
                    hovertemplate='<b>%{label}</b><br>Total Customers: %{value}<extra></extra>'
                ),
                link=dict(
                    source=[link[0] for link in links],
                    target=[link[1] for link in links],
                    value=link_values,
                    color=[f"rgba{tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" 
                           for c in link_colors],
                    customdata=link_labels,
                    hovertemplate='<b>%{customdata}</b><extra></extra>'
                ),
                arrangement='snap'
            )],
            name=f"Stage {stage}"
        )
        frames.append(frame)
    
    # Create initial figure with the first frame data
    if frames:
        initial_data = frames[0].data
    else:
        # Fallback to empty Sankey
        initial_data = [go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_products,
                color=[color_map.get(prod, "gray") for prod in all_products],
            ),
            link=dict(source=[], target=[], value=[])
        )]
    
    fig = go.Figure(
        data=initial_data,
        frames=frames,
        layout=go.Layout(
            title={
                'text': "Animated Customer Journey Flow",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            font=dict(size=16, color="white"),
            width=1000,
            height=800,
            autosize=True,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 1000, "redraw": True},
                                         "fromcurrent": True, "mode": "immediate"}]
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate"}]
                        )
                    ],
                    x=0.1,
                    y=0,
                    xanchor="right",
                    yanchor="top"
                )
            ],
            sliders=[{
                "steps": [
                    {
                        "method": "animate",
                        "label": f"Stage {i+1}",
                        "args": [[f"Stage {i+1}"], {"frame": {"duration": 300, "redraw": True},
                                                   "mode": "immediate"}]
                    }
                    for i in range(len(frames))
                ],
                "active": 0,
                "currentvalue": {"prefix": "Viewing: "},
                "x": 0.1,
                "y": 0,
                "len": 0.9,
                "xanchor": "left",
                "yanchor": "top"
            }],
            margin=dict(l=25, r=25, t=50, b=100),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
        )
    )
    
    return fig

def plot_sankey_by_starting_product(journey_df, max_paths=20, min_customers=50):
    """
    Create a dictionary of Sankey figures, one per starting product, showing the customer journeys
    for that starting point, using a consistent color mapping per product across all diagrams.
    Improved for better visualization and readability.
    
    Parameters:
    -----------
    journey_df : pd.DataFrame
        DataFrame with customer journey data. Must include a 'first_product' column.
    max_paths : int, optional
        Maximum number of journey paths to include (default is 20).
    min_customers : int, optional
        Minimum number of customers per journey path required to include that path (default is 50).
    
    Returns:
    --------
    dict
        A dictionary where keys are the starting products and values are the corresponding
        Plotly Sankey figures.
    """
    sankey_figs = {}

    # Compute a global color map for all products across the entire journey_df.
    global_products = set()
    for sequence in journey_df['sequence']:
        global_products.update(sequence.split(" → "))
    global_products = sorted(global_products)
    
    # Use brighter, more distinct colors
    color_palette = [
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', 
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
        '#AECDE8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5'
    ]
    global_color_map = {prod: color_palette[i % len(color_palette)] 
                        for i, prod in enumerate(global_products)}

    # Group journeys by the 'first_product' and create a Sankey diagram for each group.
    for first_product, group_df in journey_df.groupby('first_product'):
        # Skip groups with too few journeys
        if group_df.empty or group_df.shape[0] < min_customers:
            continue
        
        fig = plot_customer_journey_sankey(
            group_df,
            max_paths=max_paths,
            min_customers=min_customers // 2,  # Lower threshold for individual product views
            color_map=global_color_map
        )
        
        # Update layout for this specific product
        fig.update_layout(
            title={
                'text': f"Customer Journeys Starting with {first_product}",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            height=700  # Slightly smaller than the main diagram
        )
        
        sankey_figs[first_product] = fig

    return sankey_figs

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