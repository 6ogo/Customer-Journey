import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Set page config
st.set_page_config(page_title="Customer Journey Analysis", layout="wide")

# Apply minimal styling
st.markdown("""
<style>
    .main .block-container { max-width: 1200px; }
    h1, h2, h3 { color: #4da6ff; }
</style>
""", unsafe_allow_html=True)

def load_data(file_path):
    """Simple function to load CSV data with flexible separator detection"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
            
        # Try to detect the separator
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.readline()
        
        # Auto-detect the separator
        if ';' in sample:
            sep = ';'
        elif ',' in sample:
            sep = ','
        else:
            sep = '\t'  # Default to tab if neither ; nor , is found
            
        # Load the CSV
        df = pd.read_csv(file_path, sep=sep, encoding='utf-8', errors='ignore')
        
        if 'sCustomerNaturalKey' not in df.columns:
            # Try to find a suitable customer ID column
            potential_id_cols = [col for col in df.columns if 'customer' in col.lower() 
                               or 'id' in col.lower() or 'key' in col.lower()]
            
            if potential_id_cols:
                # Rename the first potential ID column to sCustomerNaturalKey
                df = df.rename(columns={potential_id_cols[0]: 'sCustomerNaturalKey'})
                st.info(f"Using '{potential_id_cols[0]}' as customer ID column")
            else:
                # Create an index-based customer ID
                df['sCustomerNaturalKey'] = [f"CUST_{i}" for i in range(len(df))]
                st.info("Created customer IDs as they were not found in data")
        
        # Create a copy of customer ID for reference
        df['customer_id'] = df['sCustomerNaturalKey']
        
        # Convert date columns to datetime
        date_cols = [col for col in df.columns if col.startswith('mFirst_')]
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        st.success(f"Loaded data with {len(df)} customers and {len(date_cols)} products")
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_sample_data(num_customers=100):
    """Create sample data when no file is available"""
    from datetime import timedelta
    
    # Create customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(1, num_customers+1)]
    
    # Define products
    products = ["Checking", "Savings", "CreditCard", "Mortgage", "Investment"]
    
    # Create dataframe
    df = pd.DataFrame()
    df['sCustomerNaturalKey'] = customer_ids
    df['customer_id'] = customer_ids
    
    # Add demographic data
    df['Age'] = np.random.randint(18, 80, size=num_customers)
    df['Woman'] = np.random.choice([0, 1], size=num_customers)
    df['Apartment'] = np.random.choice([0, 1], size=num_customers)
    
    # Create product acquisition dates
    base_date = datetime.now() - timedelta(days=365*2)  # 2 years ago
    
    # Each customer has a random chance of having each product
    for product in products:
        # Create product ownership column (first purchase date)
        col_name = f"mFirst_{product}"
        
        # Random ownership pattern
        has_product = np.random.choice([True, False], size=num_customers, p=[0.4, 0.6])
        
        # Generate random acquisition dates for customers who have the product
        dates = []
        for i in range(num_customers):
            if has_product[i]:
                # Random date between base_date and now
                days_to_add = np.random.randint(0, 365*2)
                dates.append(base_date + timedelta(days=days_to_add))
            else:
                dates.append(None)  # No date for customers without the product
        
        df[col_name] = dates
    
    st.success(f"Created sample dataset with {num_customers} customers and {len(products)} products")
    return df

def analyze_customer_journeys(df):
    """Extract customer journey data from the dataframe"""
    # Get product columns
    product_cols = [col for col in df.columns if col.startswith('mFirst_')]
    
    if not product_cols:
        st.warning("No product data found (columns starting with 'mFirst_')")
        return None, None
    
    # Create customer journey data
    journeys = []
    timeline_data = []
    
    # For each customer, create their product timeline and journey
    for _, customer in df.iterrows():
        customer_id = customer['customer_id']
        
        # Get all products and their acquisition dates
        products_owned = []
        
        for col in product_cols:
            date = customer[col]
            if pd.notna(date):
                product_name = col.replace('mFirst_', '')
                products_owned.append({
                    'product': product_name,
                    'date': date,
                    'customer_id': customer_id
                })
                
        # If customer has products, create journey
        if products_owned:
            # Sort by date
            products_owned.sort(key=lambda x: x['date'])
            
            # Add to timeline
            timeline_data.extend(products_owned)
            
            # Create journey
            product_sequence = [p['product'] for p in products_owned]
            product_sequence_str = ' → '.join(product_sequence)
            
            # Calculate journey length and duration
            journey_length = len(product_sequence)
            if journey_length > 1:
                duration_days = (products_owned[-1]['date'] - products_owned[0]['date']).days
                duration_days = max(1, duration_days)  # Avoid division by zero
            else:
                duration_days = 0
                
            journeys.append({
                'customer_id': customer_id,
                'sequence': product_sequence_str,
                'length': journey_length,
                'duration_days': duration_days,
                'first_product': product_sequence[0],
                'last_product': product_sequence[-1]
            })
    
    # Create DataFrames
    timeline_df = pd.DataFrame(timeline_data)
    journey_df = pd.DataFrame(journeys)
    
    return timeline_df, journey_df

def plot_customer_journeys(journey_df, min_customers=10, max_paths=10):
    """Create a Sankey diagram for customer journeys"""
    if journey_df is None or journey_df.empty:
        st.warning("No journey data available for visualization")
        return None
        
    # Filter to most common journeys
    journey_counts = journey_df['sequence'].value_counts()
    common_journeys = journey_counts[journey_counts >= min_customers].head(max_paths)
    
    if len(common_journeys) == 0:
        st.info("No journeys meet the minimum customer threshold. Using top journeys instead.")
        common_journeys = journey_counts.head(max_paths)
    
    if len(common_journeys) == 0:
        st.warning("No journey data to display")
        return None
    
    # Get all products in these journeys
    all_products = set()
    for sequence in common_journeys.index:
        all_products.update(sequence.split(" → "))
    products_list = sorted(list(all_products))
    
    # Create color map for products
    colors = px.colors.qualitative.Set2
    color_map = {product: colors[i % len(colors)] for i, product in enumerate(products_list)}
    
    # Create nodes and links for Sankey diagram
    nodes = []
    links = []
    
    for sequence, count in common_journeys.items():
        products = sequence.split(" → ")
        
        # Add nodes and map to indices
        for product in products:
            if product not in nodes:
                nodes.append(product)
        
        # Add links between consecutive products
        for i in range(len(products) - 1):
            source = nodes.index(products[i])
            target = nodes.index(products[i + 1])
            links.append({
                'source': source,
                'target': target,
                'value': count,
                'color': color_map[products[i]]
            })
    
    # Create Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=[color_map.get(node, "gray") for node in nodes]
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color=[f"rgba{tuple(int(c.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (0.7,)}" 
                  for c in [link['color'] for link in links]]
        )
    )])
    
    fig.update_layout(
        title="Customer Journey Flows",
        font=dict(size=14),
        height=600
    )
    
    return fig

def main():
    st.title("Customer Journey Analysis Dashboard")
    
    # Sidebar options
    with st.sidebar:
        st.header("Settings")
        file_path = st.text_input("Data File Path", "data/ABT_Score_Example.csv")
        use_sample = st.checkbox("Use Sample Data", value=True)
        
        st.divider()
        min_customers = st.slider("Min Customers per Path", 1, 50, 5)
        max_paths = st.slider("Max Paths to Show", 5, 20, 10)
    
    # Load data
    if use_sample:
        df = create_sample_data()
    else:
        df = load_data(file_path)
    
    if df is None:
        st.error("No data available. Please check your file path or use sample data.")
        return
    
    # Analyze customer journeys
    timeline_df, journey_df = analyze_customer_journeys(df)
    
    if journey_df is None or journey_df.empty:
        st.warning("Could not extract customer journeys from the data.")
        
        # Show data preview for debugging
        st.subheader("Data Preview")
        st.dataframe(df.head())
        return
    
    # Dashboard content
    tabs = st.tabs(["Overview", "Journey Analysis", "Product Analysis"])
    
    # Tab 1: Overview
    with tabs[0]:
        st.header("Customer Overview")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", f"{len(df):,}")
        with col2:
            avg_products = journey_df['length'].mean()
            st.metric("Avg Products per Customer", f"{avg_products:.1f}")
        with col3:
            multi_product = (journey_df['length'] > 1).mean() * 100
            st.metric("Multi-Product Customers", f"{multi_product:.1f}%")
        
        # Journey length distribution
        st.subheader("Journey Length Distribution")
        fig = px.histogram(
            journey_df, x='length', nbins=10,
            title="Number of Products per Customer",
            color_discrete_sequence=['#4da6ff']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Product ownership
        product_cols = [col for col in df.columns if col.startswith('mFirst_')]
        product_ownership = df[product_cols].notna().sum().sort_values(ascending=False)
        product_ownership.index = [col.replace('mFirst_', '') for col in product_ownership.index]
        
        st.subheader("Product Ownership")
        fig = px.bar(
            product_ownership,
            title="Number of Customers by Product",
            color_discrete_sequence=['#4da6ff']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Journey Analysis
    with tabs[1]:
        st.header("Customer Journey Analysis")
        
        # Sankey diagram of customer journeys
        fig = plot_customer_journeys(journey_df, min_customers, max_paths)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Common journey paths
        st.subheader("Most Common Journey Paths")
        journey_paths = journey_df['sequence'].value_counts().head(10)
        fig = px.bar(
            journey_paths,
            title="Top 10 Customer Journey Paths",
            color_discrete_sequence=['#4da6ff']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # First product analysis
        st.subheader("First Product Analysis")
        first_products = journey_df['first_product'].value_counts()
        fig = px.pie(
            values=first_products.values,
            names=first_products.index,
            title="First Product in Customer Journey"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Product Analysis
    with tabs[2]:
        st.header("Product Analysis")
        
        # Product transition matrix
        st.subheader("Product Transition Analysis")
        
        # Create transition matrix
        transitions = []
        for sequence in journey_df['sequence']:
            products = sequence.split(" → ")
            if len(products) > 1:
                for i in range(len(products) - 1):
                    transitions.append({
                        'from_product': products[i],
                        'to_product': products[i + 1]
                    })
        
        if transitions:
            transition_df = pd.DataFrame(transitions)
            transition_matrix = pd.crosstab(
                transition_df['from_product'], 
                transition_df['to_product'], 
                normalize='index'
            )
            
            # Display transition matrix
            st.dataframe(transition_matrix.style.format("{:.1%}").background_gradient(cmap='Blues'))
            
            # Product selection for next-product analysis
            st.subheader("Next Product Analysis")
            products = sorted(list(set(transition_df['from_product'])))
            if products:
                selected_product = st.selectbox("Select Product", products)
                
                if selected_product in transition_matrix.index:
                    next_products = transition_matrix.loc[selected_product].sort_values(ascending=False)
                    
                    st.write(f"Top products purchased after {selected_product}:")
                    fig = px.bar(
                        next_products.head(5),
                        title=f"Products Purchased After {selected_product}",
                        color_discrete_sequence=['#4da6ff']
                    )
                    fig.update_layout(yaxis_tickformat='.0%')
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough multi-product journeys to analyze transitions")

if __name__ == "__main__":
    main()