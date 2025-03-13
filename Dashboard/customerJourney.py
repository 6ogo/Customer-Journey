import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Import helper functions
from utils import (
    load_csv_file,
    preprocess_data,
    analyze_product_sequence,
    analyze_lifecycle_stages,
    analyze_journey_patterns,
    analyze_churn_risk,
    create_product_timeline,
    plot_lifecycle_analysis
)

# Import our enhanced visualization functions
from improved_sankey import (
    plot_customer_journey_sankey,
    plot_sankey_by_starting_product,
    plot_animated_journey_sankey
)

# Set page config with wider layout
st.set_page_config(
    page_title="Customer Journey Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styling
st.markdown("""
<style>
    .main .block-container {
        max-width: 1200px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    .stPlotlyChart {
        display: flex;
        justify-content: center;
    }
    h1, h2, h3 {
        color: #4da6ff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1f1f1f;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4da6ff !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_preprocess_data(file_path):
    try:
        combined_df = load_csv_file(file_path)
        if combined_df is None or combined_df.empty:
            st.error("No data loaded from file")
            return None, None, None
        if 'sCustomerNaturalKey' in combined_df.columns:
            combined_df = combined_df.set_index('sCustomerNaturalKey')
        combined_df = preprocess_data(combined_df)
        timeline_df, journey_df = analyze_product_sequence(combined_df)
        return combined_df, timeline_df, journey_df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

@st.cache_data
def create_product_transition_matrix(journey_df):
    """Create a transition matrix showing product purchase sequences"""
    if journey_df.empty:
        return pd.DataFrame()
        
    transitions = []
    
    for sequence in journey_df['sequence']:
        products = sequence.split(' → ')
        if len(products) > 1:
            for i in range(len(products) - 1):
                transitions.append({
                    'from_product': products[i],
                    'to_product': products[i + 1]
                })
    
    if not transitions:
        return pd.DataFrame()
        
    transition_df = pd.DataFrame(transitions)
    transition_matrix = pd.crosstab(
        transition_df['from_product'], 
        transition_df['to_product'], 
        normalize='index'
    )
    
    return transition_matrix

@st.cache_data
def analyze_product_demographics(combined_df, product):
    """Analyze demographics for a specific product with better error handling"""
    if combined_df is None or combined_df.empty or not product:
        return {
            'age_mean': 0,
            'age_median': 0,
            'pct_women': 0,
            'pct_apartment': 0,
            'total_customers': 0
        }
        
    # Make sure the column exists before trying to use it
    product_col = f'mFirst_{product}'
    if product_col not in combined_df.columns:
        print(f"Warning: Column {product_col} not found in data")
        return {
            'age_mean': 0,
            'age_median': 0,
            'pct_women': 0,
            'pct_apartment': 0,
            'total_customers': 0
        }
        
    product_customers = combined_df[combined_df[product_col].notna()]
    
    if product_customers.empty:
        return {
            'age_mean': 0,
            'age_median': 0,
            'pct_women': 0,
            'pct_apartment': 0,
            'total_customers': 0
        }
    
    demographics = {
        'age_mean': product_customers['Age'].mean(),
        'age_median': product_customers['Age'].median(),
        'pct_women': (product_customers['Woman'] == 1).mean() * 100,
        'pct_apartment': (product_customers['Apartment'] == 1).mean() * 100,
        'total_customers': len(product_customers)
    }
    
    return demographics

def main():
    st.title("Customer Journey Analysis Dashboard")
    
    # Add configuration options in sidebar
    with st.sidebar:
        st.header("Visualization Settings")
        min_customers = st.slider("Minimum Customers per Path", 10, 200, 30)
        max_paths = st.slider("Maximum Paths to Show", 5, 50, 15)
        show_animation = st.checkbox("Show Animated Journey Flow", value=True)
        st.divider()
        file_path = st.text_input("Data File Path", "./Data/customer_data.csv")
    
    try:
        # Load data with better error handling
        data = load_and_preprocess_data(file_path)
        if data is None or any(x is None for x in data):
            st.error("Failed to load required data. Please check your data file and try again.")
            return
            
        combined_df, timeline_df, journey_df = data
        
        # Additional validation for empty DataFrames
        if combined_df.empty:
            st.error("The main customer dataset is empty. Please check your data.")
            return
            
        # Check if timeline_df and journey_df are empty
        if timeline_df.empty:
            st.warning("No timeline data was created. This may limit visualizations.")
            
        if journey_df.empty:
            st.warning("No journey data was created. Most visualizations will be empty.")
            
        # Compute metrics safely with error handling
        total_customers = combined_df['sCustomerNaturalKey'].nunique()
        multi_product_customers = journey_df.shape[0] if not journey_df.empty else 0
        multi_product_pct = multi_product_customers / total_customers if total_customers > 0 else 0
    
        # Create tabs with improved styling
        tabs = st.tabs([
            "📊 Overview & Journey Analysis", 
            "🏦 Product Analysis", 
            "🔄 Animated Journey Flow",
            "💼 CRM & Recommendations"
        ])
        
        # Tab 1: Overview & Journey Analysis
        with tabs[0]:
            st.header("Customer Journey Overview")
            
            # Key metrics in a nicer format
            avg_journey_length = journey_df['length'].mean() if not journey_df.empty else 0
            
            metrics_cols = st.columns(3)
            with metrics_cols[0]:
                st.metric("Total Customers", f"{total_customers:,}")
            with metrics_cols[1]:
                st.metric("Avg Journey Length", f"{avg_journey_length:.2f} products")
            with metrics_cols[2]:
                st.metric("Multi-product %", f"{multi_product_pct:.1%}")
            
            st.divider()
            
            # Journey Length Distribution - improved styling
            st.subheader("Journey Length Distribution")
            fig_journey = px.histogram(
                journey_df, 
                x='length',
                nbins=20,
                title="Distribution of Customer Journey Lengths",
                color_discrete_sequence=['#4da6ff'],
                opacity=0.8
            )
            fig_journey.update_layout(
                xaxis_title="Number of Products",
                yaxis_title="Number of Customers",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_journey, use_container_width=True)
            
            st.divider()

            # Main Sankey Diagram - improved centering and clarity
            st.subheader("Overall Customer Journey Flows")
            fig_sankey = plot_customer_journey_sankey(journey_df, max_paths=max_paths, min_customers=min_customers)
            st.plotly_chart(fig_sankey, use_container_width=True)
            
            st.divider()
            
            # Common Journey Paths
            st.subheader("Most Common Journey Paths")
            journey_paths = journey_df['sequence'].value_counts().head(10)
            fig_paths = px.bar(
                journey_paths,
                title="Top 10 Customer Journey Paths",
                color_discrete_sequence=['#4da6ff']
            )
            fig_paths.update_layout(
                xaxis_title="Journey Path",
                yaxis_title="Number of Customers",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_paths, use_container_width=True)
        
        # Tab 2: Product Analysis
        with tabs[1]:
            st.header("Product Analysis")
            
            # Product ownership analysis
            product_cols = [col for col in combined_df.columns if col.startswith('mFirst_')]
            product_ownership = combined_df[product_cols].notna().sum().sort_values(ascending=False)
            product_ownership.index = [col.replace('mFirst_', '') for col in product_ownership.index]
            
            st.subheader("Product Ownership")
            fig_ownership = px.bar(
                product_ownership,
                title="Number of Customers by Product",
                color_discrete_sequence=['#4da6ff']
            )
            fig_ownership.update_layout(
                xaxis_title="Product",
                yaxis_title="Number of Customers",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_ownership, use_container_width=True)
            
            st.divider()
            
            # Product correlation heatmap with improved styling
            st.subheader("Product Correlations")
            product_cols = [col for col in combined_df.columns if col.startswith('mFirst_')]
            if not product_cols:
                st.warning("No product columns found in data")
            else:
                # Convert date columns to boolean for correlation
                product_data = combined_df[product_cols].notna().astype(int)
                
                # Ensure we have enough non-zero values
                if product_data.sum().sum() == 0:
                    st.warning("No product data available for correlation analysis")
                else:
                    product_cols_renamed = [col.replace('mFirst_', '') for col in product_cols]
                    corr_matrix = product_data.corr()
                    corr_matrix.index = product_cols_renamed
                    corr_matrix.columns = product_cols_renamed
                    
                    fig_corr = px.imshow(
                        corr_matrix,
                        title="Product Correlation Heatmap",
                        color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1
                    )
                    fig_corr.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            st.divider()
            
            # Product demographics with improved styling
            st.subheader("Product Demographics")
            selected_product = st.selectbox(
                "Select Product for Demographic Analysis",
                [col.replace('mFirst_', '') for col in product_cols]
            )
            
            demographics = analyze_product_demographics(combined_df, selected_product)
            
            demo_cols = st.columns(4)
            with demo_cols[0]:
                st.metric("Average Age", f"{demographics['age_mean']:.1f}")
            with demo_cols[1]:
                st.metric("Median Age", f"{demographics['age_median']:.1f}")
            with demo_cols[2]:
                st.metric("Women Customers", f"{demographics['pct_women']:.1f}%")
            with demo_cols[3]:
                st.metric("Apartment Dwellers", f"{demographics['pct_apartment']:.1f}%")
                
            st.metric("Total Customers", f"{demographics['total_customers']:,}")
        
        # Tab 3: Animated Journey Flow
        with tabs[2]:
            st.header("Animated Customer Journey Flow")
            
            if show_animation:
                st.info("This visualization breaks down customer journeys step by step. Use the play button or slider to see how customers move between products over time.")
                
                # Create animated Sankey diagram
                fig_animated = plot_animated_journey_sankey(journey_df, max_paths=max_paths, min_customers=min_customers)
                st.plotly_chart(fig_animated, use_container_width=True)
                
                st.caption("Note: Animation shows the progressive flow of customers between products. Each stage represents an additional step in the customer journey.")
            else:
                st.subheader("Customer Journey Flows by starting product")
                # Create multiple Sankey diagrams grouped by the first product in each journey
                sankey_figs = plot_sankey_by_starting_product(journey_df, max_paths=max_paths, min_customers=min_customers)

                # Show each diagram in its own subheader
                if sankey_figs:
                    for start_prod, fig in sankey_figs.items():
                        st.subheader(f"Customer Journeys Starting with {start_prod}")
                        st.plotly_chart(fig, use_container_width=True)
                        st.divider()
                else:
                    st.warning("No product-specific journey flows meet the current threshold criteria. Try lowering the minimum customers per path.")
        
        # Tab 4: CRM & Recommendations
        with tabs[3]:
            st.header("CRM Recommendations")
            
            # Product selection for recommendations
            selected_product = st.selectbox(
                "Select Product for Recommendations",
                [col.replace('mFirst_', '') for col in product_cols],
                key="crm_product_select"
            )
            
            # Create transition matrix
            transition_matrix = create_product_transition_matrix(journey_df)
            
            # Get recommendations
            if selected_product in transition_matrix.index:
                next_products = transition_matrix.loc[selected_product].sort_values(ascending=False)
                
                st.subheader("Product Recommendations")
                cols = st.columns([3, 2])
                
                with cols[0]:
                    st.write("#### Top Products Purchased After")
                    
                    # Create a nicer visualization of the transition probabilities
                    fig_next = px.bar(
                        next_products.head(5),
                        orientation='h',
                        title=f"Products Customers Buy After {selected_product}",
                        labels={'value': 'Probability', 'index': 'Product'},
                        color_discrete_sequence=['#4da6ff']
                    )
                    fig_next.update_layout(
                        xaxis_title="Transition Probability",
                        yaxis_title="Next Product",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=40, r=40, t=40, b=20),
                        xaxis=dict(tickformat='.0%')
                    )
                    st.plotly_chart(fig_next)
                    
                    # Show raw probability table below the chart
                    st.write("Probability Table:")
                    st.dataframe(
                        next_products.head(5).to_frame('Probability').style.format("{:.1%}").background_gradient(cmap='Blues'),
                        use_container_width=True
                    )
                
                with cols[1]:
                    demographics = analyze_product_demographics(combined_df, selected_product)
                    st.write("#### Target Customer Profile")
                    
                    # Create a spider/radar chart for demographics
                    categories = ['Age', 'Women %', 'Apartment %']
                    values = [
                        min(100, demographics['age_mean'] / 100 * 100),  # Scale age to 0-100
                        demographics['pct_women'],
                        demographics['pct_apartment']
                    ]
                    
                    fig_profile = go.Figure()
                    
                    fig_profile.add_trace(go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=selected_product,
                        line_color='#4da6ff',
                        fillcolor='rgba(77, 166, 255, 0.3)'
                    ))
                    
                    fig_profile.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )
                        ),
                        showlegend=False,
                        title="Key Demographics",
                        font=dict(color="white"),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                    )
                    
                    st.plotly_chart(fig_profile, use_container_width=True)
                    
                    # Show exact values
                    st.write(f"• Average Age: **{demographics['age_mean']:.1f}**")
                    st.write(f"• Median Age: **{demographics['age_median']:.1f}**")
                    st.write(f"• Women: **{demographics['pct_women']:.1f}%**")
                    st.write(f"• Apartment Dwellers: **{demographics['pct_apartment']:.1f}%**")
                    st.write(f"• Total Customers: **{demographics['total_customers']:,}**")
                
                st.divider()
                
                # Lifecycle stage analysis with improved visuals
                st.subheader("Customer Lifecycle Analysis")
                customer_data = combined_df[combined_df[f'mFirst_{selected_product}'].notna()]
                lifecycle_data = analyze_lifecycle_stages(journey_df)
                
                # Create tabs for different lifecycle visualizations
                lifecycle_tabs = st.tabs(["Adoption Rate", "Journey Summary"])
                
                with lifecycle_tabs[0]:
                    fig_lifecycle = plot_lifecycle_analysis(lifecycle_data)
                    if fig_lifecycle:
                        # Enhance the styling
                        fig_lifecycle.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=40, r=40, t=40, b=40)
                        )
                        st.plotly_chart(fig_lifecycle, use_container_width=True)
                    else:
                        st.info("No lifecycle data available for visualization")
                
                with lifecycle_tabs[1]:
                    # Journey patterns summary
                    journey_patterns = analyze_journey_patterns(journey_df)
                    
                    # Journey stats
                    stats_cols = st.columns(4)
                    with stats_cols[0]:
                        st.metric("Total Customers", f"{journey_patterns['journey_stats']['total_customers']:,}")
                    with stats_cols[1]:
                        st.metric("Avg Products", f"{journey_patterns['journey_stats']['avg_products']:.1f}")
                    with stats_cols[2]:
                        st.metric("Single Product %", f"{journey_patterns['journey_segments']['single_product']:.1%}")
                    with stats_cols[3]:
                        st.metric("Long Journeys %", f"{journey_patterns['journey_segments']['long_journey']:.1%}")
                    
                    # Show common first products
                    common_first = journey_patterns['journey_stats']['common_first']
                    if not isinstance(common_first, pd.Series) or common_first.empty:
                        st.info("No common first product data available")
                    else:
                        st.write("#### Common First Products")
                        fig_first = px.pie(
                            values=common_first.values,
                            names=common_first.index,
                            title="Most Common First Products",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        fig_first.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='white'),
                            margin=dict(l=20, r=20, t=40, b=20)
                        )
                        st.plotly_chart(fig_first, use_container_width=True)
                
                st.divider()
                
                # Churn Risk Analysis with improved visuals
                st.subheader("Churn Risk Analysis")
                risk_data = analyze_churn_risk(journey_df, combined_df, timeline_df)
                
                if not risk_data.empty:
                    # Create a more informative visualization of churn risk
                    fig_risk = px.histogram(
                        risk_data,
                        x='days_since_last_product',
                        nbins=50,
                        title="Days Since Last Product Purchase",
                        color_discrete_sequence=['#ff6666']  # Red for risk
                    )
                    
                    # Add vertical reference lines for risk categories
                    fig_risk.add_vline(x=180, line_dash="dash", line_color="yellow",
                                    annotation_text="Medium Risk (6+ months)")
                    fig_risk.add_vline(x=365, line_dash="dash", line_color="red",
                                    annotation_text="High Risk (1+ year)")
                    
                    fig_risk.update_layout(
                        xaxis_title="Days Since Last Purchase",
                        yaxis_title="Number of Customers",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_risk, use_container_width=True)
                    
                    # Risk metrics
                    high_risk = risk_data[risk_data['days_since_last_product'] > 365]
                    medium_risk = risk_data[(risk_data['days_since_last_product'] > 180) & 
                                        (risk_data['days_since_last_product'] <= 365)]
                    
                    risk_cols = st.columns(3)
                    with risk_cols[0]:
                        st.metric("High Risk Customers", f"{len(high_risk):,}",
                                delta=f"{len(high_risk)/len(risk_data):.1%} of total",
                                delta_color="inverse")
                    with risk_cols[1]:
                        st.metric("Medium Risk Customers", f"{len(medium_risk):,}",
                                delta=f"{len(medium_risk)/len(risk_data):.1%} of total",
                                delta_color="inverse")
                    with risk_cols[2]:
                        avg_days = risk_data['days_since_last_product'].mean()
                        st.metric("Avg Days Since Purchase", f"{avg_days:.0f}")
                    
                    # High-risk customers detailed analysis
                    if not high_risk.empty:
                        with st.expander("High-Risk Customer Analysis", expanded=True):
                            st.warning(f"⚠️ Found {len(high_risk):,} high-risk customers (no purchase in >1 year)")
                            
                            # Show high-risk customer demographics
                            high_risk_customers = combined_df[combined_df.index.isin(high_risk.index)]
                            high_risk_demographics = analyze_product_demographics(high_risk_customers, selected_product)
                            
                            risk_demo_cols = st.columns(2)
                            
                            with risk_demo_cols[0]:
                                st.write("##### High-Risk Customer Profile")
                                profile_cols = st.columns(2)
                                with profile_cols[0]:
                                    st.metric("Average Age", f"{high_risk_demographics['age_mean']:.1f}")
                                    st.metric("Women %", f"{high_risk_demographics['pct_women']:.1f}%")
                                with profile_cols[1]:
                                    st.metric("Total Customers", high_risk_demographics['total_customers'])
                                    st.metric("Apartment %", f"{high_risk_demographics['pct_apartment']:.1f}%")
                            
                            with risk_demo_cols[1]:
                                # Compare high-risk customers to overall customer base
                                st.write("##### Comparison to Overall Customer Base")
                                
                                # Create comparison metrics
                                age_diff = high_risk_demographics['age_mean'] - demographics['age_mean']
                                women_diff = high_risk_demographics['pct_women'] - demographics['pct_women']
                                apt_diff = high_risk_demographics['pct_apartment'] - demographics['pct_apartment']
                                
                                comp_cols = st.columns(3)
                                with comp_cols[0]:
                                    st.metric("Age Difference", f"{age_diff:+.1f} years",
                                            delta_color="off")
                                with comp_cols[1]:
                                    st.metric("Women % Difference", f"{women_diff:+.1f}%",
                                            delta_color="off")
                                with comp_cols[2]:
                                    st.metric("Apartment % Difference", f"{apt_diff:+.1f}%",
                                            delta_color="off")
            else:
                st.info(f"No transition data available for {selected_product}")
                
                # Still show basic product information
                demographics = analyze_product_demographics(combined_df, selected_product)
                if demographics['total_customers'] > 0:
                    st.write(f"### {selected_product} Product Customer Profile")
                    
                    profile_cols = st.columns(4)
                    with profile_cols[0]:
                        st.metric("Total Customers", f"{demographics['total_customers']:,}")
                    with profile_cols[1]:
                        st.metric("Average Age", f"{demographics['age_mean']:.1f}")
                    with profile_cols[2]:
                        st.metric("Women Customers", f"{demographics['pct_women']:.1f}%")
                    with profile_cols[3]:
                        st.metric("Apartment Dwellers", f"{demographics['pct_apartment']:.1f}%")
                    
                    st.info("This product doesn't have sufficient journey data to generate recommendations. This could be because it's frequently the last product in customer journeys or because transitions from this product are too diverse to create meaningful patterns.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your data file and try again.")
        # Add file information for debugging
        st.info(f"File path: {file_path}")
        st.info("Check if the file exists and has the correct format")
        
if __name__ == "__main__":
    try:
        main()
        
        # Add footer
        st.markdown("---")
        st.markdown(
            """
            <div style="text-align: center; color: #888;">
                <p>Customer Journey Analysis Dashboard • Optimized Version</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.info("Please check your data file and try again. If the problem persists, contact support.")