import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

def plot_customer_journey_sankey(journey_df, max_paths=20, min_customers=50, color_map=None):
    """Create an enhanced Sankey diagram for customer journeys"""
    # Add validation
    if journey_df is None or journey_df.empty or 'sequence' not in journey_df.columns:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No journey data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(
            title="Customer Journey Sankey Diagram",
            font=dict(color="white"),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            height=600
        )
        return fig

    # Get sequences with their counts
    try:
        sequence_counts = journey_df['sequence'].value_counts()
        sequence_counts = sequence_counts[sequence_counts >= min_customers].head(max_paths)
        if len(sequence_counts) == 0:
            sequence_counts = journey_df['sequence'].value_counts().head(max_paths)
    except Exception as e:
        print(f"Error processing sequences: {str(e)}")
        sequence_counts = pd.Series()
    
    if len(sequence_counts) == 0:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No journey paths meet the criteria",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(
            title="Customer Journey Sankey Diagram",
            font=dict(color="white"),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            height=600
        )
        return fig

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
    if journey_df is None or journey_df.empty or 'sequence' not in journey_df.columns:
        # Return empty figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No journey data available for animation",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(
            title="Animated Customer Journey Flow",
            font=dict(color="white"),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            height=600
        )
        return fig
    
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