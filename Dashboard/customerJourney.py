import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Import helper functions
from utils import (
    load_abt_files,
    preprocess_data,
    analyze_product_sequence,
    analyze_lifecycle_stages,
    analyze_journey_patterns,
    analyze_churn_risk
)

# Set page config
st.set_page_config(page_title="Customer Journey Analysis", layout="wide")

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess all data"""
    combined_df = load_abt_files()
    combined_df = preprocess_data(combined_df)
    timeline_df, journey_df = analyze_product_sequence(combined_df)
    return combined_df, timeline_df, journey_df

def create_product_transition_matrix(journey_df):
    """Create a transition matrix showing product purchase sequences"""
    transitions = []
    
    for sequence in journey_df['sequence']:
        products = sequence.split(' → ')
        if len(products) > 1:
            for i in range(len(products) - 1):
                transitions.append({
                    'from_product': products[i],
                    'to_product': products[i + 1]
                })
    
    transition_df = pd.DataFrame(transitions)
    transition_matrix = pd.crosstab(
        transition_df['from_product'], 
        transition_df['to_product'], 
        normalize='index'
    )
    
    return transition_matrix

def analyze_product_demographics(combined_df, product):
    """Analyze demographics for a specific product"""
    product_customers = combined_df[combined_df[f'Have_{product}'] == 1]
    
    demographics = {
        'age_mean': product_customers['Age'].mean(),
        'age_median': product_customers['Age'].median(),
        'pct_women': (product_customers['Woman'] == 1).mean() * 100,
        'pct_apartment': (product_customers['Apartment'] == 1).mean() * 100,
        'total_customers': len(product_customers),
        'lifestyle_distribution': product_customers['LifestyleGroupCode'].value_counts()
    }
    
    return demographics

def main():
    st.title("Customer Journey Analysis Dashboard")
    
    # Load data
    try:
        combined_df, timeline_df, journey_df = load_and_preprocess_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.write("Please ensure your data files are in the correct location and format.")
        return
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs([
        "Overview & Journey Analysis", 
        "Product Analysis", 
        "CRM & Recommendations"
    ])
    
    # Tab 1: Overview & Journey Analysis
    with tab1:
        st.header("Customer Journey Overview")
        
        # Key metrics
        patterns = analyze_journey_patterns(journey_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Customers", f"{patterns['journey_stats']['total_customers']:,}")
        with col2:
            st.metric("Avg Journey Length", f"{patterns['journey_stats']['avg_products']:.2f} products")
        with col3:
            st.metric("Avg Journey Duration", f"{patterns['journey_stats']['avg_duration']:.0f} days")
        with col4:
            st.metric("Multi-product Customers", 
                     f"{(1 - patterns['journey_segments']['single_product']):.1%}")
        
        # Journey Length Distribution
        st.subheader("Journey Length Distribution")
        fig_journey = px.histogram(
            journey_df, 
            x='length',
            nbins=20,
            title="Distribution of Customer Journey Lengths"
        )
        st.plotly_chart(fig_journey, use_container_width=True)
        
        # Product Adoption Timeline
        st.subheader("Product Adoption Timeline")
        fig_timeline = px.scatter(
            timeline_df,
            x='acquisition_date',
            y='product',
            color='product',
            title="Product Adoption Over Time"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Common Journey Paths
        st.subheader("Most Common Journey Paths")
        journey_paths = journey_df['sequence'].value_counts().head(10)
        fig_paths = px.bar(
            journey_paths,
            title="Top 10 Customer Journey Paths"
        )
        st.plotly_chart(fig_paths, use_container_width=True)
    
    # Tab 2: Product Analysis
    with tab2:
        st.header("Product Analysis")
        
        # Product ownership analysis
        have_cols = [col for col in combined_df.columns if col.startswith('Have_')]
        product_ownership = combined_df[have_cols].sum().sort_values(ascending=False)
        
        st.subheader("Product Ownership")
        fig_ownership = px.bar(
            product_ownership,
            title="Number of Customers by Product"
        )
        st.plotly_chart(fig_ownership, use_container_width=True)
        
        # Product correlation heatmap
        st.subheader("Product Correlations")
        corr_matrix = combined_df[have_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            title="Product Correlation Heatmap",
            color_continuous_scale="RdBu"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Product demographics
        st.subheader("Product Demographics")
        selected_product = st.selectbox(
            "Select Product for Demographic Analysis",
            [col.replace('Have_', '') for col in have_cols]
        )
        
        demographics = analyze_product_demographics(combined_df, selected_product)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Age", f"{demographics['age_mean']:.1f}")
        with col2:
            st.metric("Women Customers", f"{demographics['pct_women']:.1f}%")
        with col3:
            st.metric("Apartment Dwellers", f"{demographics['pct_apartment']:.1f}%")
    
    # Tab 3: CRM & Recommendations
    with tab3:
        st.header("CRM Recommendations")
        
        # Product selection for recommendations
        selected_product = st.selectbox(
            "Select Product for Recommendations",
            [col.replace('Have_', '') for col in have_cols],
            key="crm_product_select"
        )
        
        # Create transition matrix
        transition_matrix = create_product_transition_matrix(journey_df)
        
        # Get recommendations
        if selected_product in transition_matrix.index:
            next_products = transition_matrix.loc[selected_product].sort_values(ascending=False)
            
            st.subheader("Product Recommendations")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Top Products Purchased After")
                st.dataframe(next_products.head().to_frame('Probability'))
            
            with col2:
                demographics = analyze_product_demographics(combined_df, selected_product)
                st.write("Target Customer Profile")
                demographics = analyze_product_demographics(combined_df, selected_product)
                st.write("Target Customer Profile")
                st.write(f"- Average Age: {demographics['age_mean']:.1f}")
                st.write(f"- Median Age: {demographics['age_median']:.1f}")
                st.write(f"- Women: {demographics['pct_women']:.1f}%")
                st.write(f"- Apartment Dwellers: {demographics['pct_apartment']:.1f}%")
                st.write("\nLifestyle Distribution:")
                st.dataframe(demographics['lifestyle_distribution'])
            
            # Lifecycle stage analysis
            st.subheader("Customer Lifecycle Analysis")
            customer_data = combined_df[combined_df[f'Have_{selected_product}'] == 1]
            lifecycle_data = analyze_lifecycle_stages(journey_df, customer_data)
            
            if not lifecycle_data.empty:
                fig_lifecycle = px.box(
                    lifecycle_data,
                    x='stage',
                    y='adoption_rate',
                    title="Adoption Rate by Lifecycle Stage"
                )
                st.plotly_chart(fig_lifecycle, use_container_width=True)
            
            # Churn Risk Analysis
            st.subheader("Churn Risk Analysis")
            risk_data = analyze_churn_risk(journey_df, combined_df, timeline_df)
            
            if not risk_data.empty:
                fig_risk = px.histogram(
                    risk_data,
                    x='days_since_last_product',
                    title="Days Since Last Product Purchase"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
                
                # High-risk customers
                high_risk = risk_data[risk_data['days_since_last_product'] > 365]
                if not high_risk.empty:
                    st.warning(f"Found {len(high_risk)} high-risk customers (no purchase in >1 year)")
                    
                    # Show high-risk customer demographics
                    high_risk_customers = combined_df[combined_df.index.isin(high_risk.index)]
                    high_risk_demographics = analyze_product_demographics(high_risk_customers, selected_product)
                    
                    st.write("High-Risk Customer Profile:")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average Age", f"{high_risk_demographics['age_mean']:.1f}")
                        st.metric("Women %", f"{high_risk_demographics['pct_women']:.1f}%")
                    with col2:
                        st.metric("Total Customers", high_risk_demographics['total_customers'])
                        st.metric("Apartment %", f"{high_risk_demographics['pct_apartment']:.1f}%")

if __name__ == "__main__":
    main()