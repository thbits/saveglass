"""
Utility functions for data generation and plotting
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime, timedelta
import random


def generate_sample_data(prompt: str) -> pd.DataFrame:
    """
    Generate sample data based on the user's request.
    
    Args:
        prompt: User's request/prompt
        
    Returns:
        DataFrame with sample data
    """
    prompt_lower = prompt.lower()
    
    # Determine data type based on keywords in prompt
    if any(keyword in prompt_lower for keyword in ["sales", "revenue", "profit"]):
        return _generate_sales_data()
    elif any(keyword in prompt_lower for keyword in ["stock", "price", "market"]):
        return _generate_stock_data()
    elif any(keyword in prompt_lower for keyword in ["weather", "temperature", "climate"]):
        return _generate_weather_data()
    elif any(keyword in prompt_lower for keyword in ["user", "customer", "demographic"]):
        return _generate_user_data()
    elif any(keyword in prompt_lower for keyword in ["scatter", "correlation"]):
        return _generate_scatter_data()
    else:
        return _generate_generic_data()


def _generate_sales_data() -> pd.DataFrame:
    """Generate sample sales data."""
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    
    data = []
    for month in months:
        for product in products:
            sales = random.randint(1000, 5000)
            data.append({
                'Month': month,
                'Product': product,
                'Sales': sales,
                'Revenue': sales * random.uniform(10, 50)
            })
    
    return pd.DataFrame(data)


def _generate_stock_data() -> pd.DataFrame:
    """Generate sample stock price data."""
    start_date = datetime.now() - timedelta(days=30)
    dates = [start_date + timedelta(days=i) for i in range(30)]
    
    # Simple random walk for stock price
    price = 100
    prices = []
    for _ in dates:
        price += random.uniform(-5, 5)
        price = max(50, min(150, price))  # Keep price in reasonable range
        prices.append(price)
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices,
        'Volume': [random.randint(10000, 100000) for _ in dates]
    })


def _generate_weather_data() -> pd.DataFrame:
    """Generate sample weather data."""
    days = 30
    dates = [datetime.now() - timedelta(days=i) for i in range(days)]
    dates.reverse()
    
    return pd.DataFrame({
        'Date': dates,
        'Temperature': [random.uniform(15, 35) for _ in range(days)],
        'Humidity': [random.uniform(30, 90) for _ in range(days)],
        'Precipitation': [random.uniform(0, 10) for _ in range(days)]
    })


def _generate_user_data() -> pd.DataFrame:
    """Generate sample user demographic data."""
    age_groups = ['18-25', '26-35', '36-45', '46-55', '56+']
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    data = []
    for age_group in age_groups:
        for region in regions:
            count = random.randint(50, 500)
            data.append({
                'Age Group': age_group,
                'Region': region,
                'Count': count,
                'Satisfaction': random.uniform(3.0, 5.0)
            })
    
    return pd.DataFrame(data)


def _generate_scatter_data() -> pd.DataFrame:
    """Generate sample data for scatter plots."""
    n_points = 50
    x = np.random.normal(0, 1, n_points)
    y = 2 * x + np.random.normal(0, 0.5, n_points)
    
    return pd.DataFrame({
        'X': x,
        'Y': y,
        'Category': [random.choice(['A', 'B', 'C']) for _ in range(n_points)]
    })


def _generate_generic_data() -> pd.DataFrame:
    """Generate generic sample data."""
    categories = ['Category A', 'Category B', 'Category C', 'Category D']
    return pd.DataFrame({
        'Category': categories,
        'Value': [random.randint(10, 100) for _ in categories],
        'Score': [random.uniform(0, 10) for _ in categories]
    })


def create_plot(data: pd.DataFrame, prompt: str) -> Optional[go.Figure]:
    """
    Create a plot based on the data and user prompt.
    
    Args:
        data: DataFrame with the data to plot
        prompt: User's request/prompt
        
    Returns:
        Plotly figure or None if no suitable plot can be created
    """
    if data.empty:
        return None
    
    prompt_lower = prompt.lower()
    
    try:
        # Determine plot type based on prompt and data
        if any(keyword in prompt_lower for keyword in ["line", "trend", "time"]) and 'Date' in data.columns:
            return _create_line_plot(data, prompt)
        elif any(keyword in prompt_lower for keyword in ["bar", "sales", "comparison"]):
            return _create_bar_plot(data, prompt)
        elif any(keyword in prompt_lower for keyword in ["scatter", "correlation"]):
            return _create_scatter_plot(data, prompt)
        elif any(keyword in prompt_lower for keyword in ["pie", "proportion", "distribution"]):
            return _create_pie_plot(data, prompt)
        elif any(keyword in prompt_lower for keyword in ["histogram", "frequency"]):
            return _create_histogram(data, prompt)
        else:
            # Default to bar plot for categorical data, line plot for time series
            if 'Date' in data.columns:
                return _create_line_plot(data, prompt)
            else:
                return _create_bar_plot(data, prompt)
    
    except Exception as e:
        print(f"Error creating plot: {e}")
        return None


def _create_line_plot(data: pd.DataFrame, prompt: str) -> go.Figure:
    """Create a line plot."""
    date_col = 'Date'
    value_cols = [col for col in data.columns if col not in [date_col] and data[col].dtype in ['int64', 'float64']]
    
    if not value_cols:
        return None
    
    fig = go.Figure()
    
    for col in value_cols[:3]:  # Limit to 3 lines for readability
        fig.add_trace(go.Scatter(
            x=data[date_col],
            y=data[col],
            mode='lines+markers',
            name=col
        ))
    
    fig.update_layout(
        title=f"Time Series: {', '.join(value_cols[:3])}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode='x unified'
    )
    
    return fig


def _create_bar_plot(data: pd.DataFrame, prompt: str) -> go.Figure:
    """Create a bar plot."""
    # Find categorical and numerical columns
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    if not cat_cols or not num_cols:
        return None
    
    # Use first categorical and first numerical column
    x_col = cat_cols[0]
    y_col = num_cols[0]
    
    # Group by categorical column if needed
    if len(data) > 20:
        plot_data = data.groupby(x_col)[y_col].sum().reset_index()
    else:
        plot_data = data[[x_col, y_col]].drop_duplicates()
    
    fig = px.bar(
        plot_data,
        x=x_col,
        y=y_col,
        title=f"{y_col} by {x_col}"
    )
    
    fig.update_layout(showlegend=False)
    return fig


def _create_scatter_plot(data: pd.DataFrame, prompt: str) -> go.Figure:
    """Create a scatter plot."""
    num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    if len(num_cols) < 2:
        return None
    
    x_col, y_col = num_cols[0], num_cols[1]
    
    # Check if there's a categorical column for color coding
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    color_col = cat_cols[0] if cat_cols else None
    
    fig = px.scatter(
        data,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f"{y_col} vs {x_col}"
    )
    
    return fig


def _create_pie_plot(data: pd.DataFrame, prompt: str) -> go.Figure:
    """Create a pie chart."""
    cat_cols = [col for col in data.columns if data[col].dtype == 'object']
    num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    if not cat_cols or not num_cols:
        return None
    
    # Group by first categorical column
    pie_data = data.groupby(cat_cols[0])[num_cols[0]].sum().reset_index()
    
    fig = px.pie(
        pie_data,
        values=num_cols[0],
        names=cat_cols[0],
        title=f"Distribution of {num_cols[0]} by {cat_cols[0]}"
    )
    
    return fig


def _create_histogram(data: pd.DataFrame, prompt: str) -> go.Figure:
    """Create a histogram."""
    num_cols = [col for col in data.columns if data[col].dtype in ['int64', 'float64']]
    
    if not num_cols:
        return None
    
    col = num_cols[0]
    
    fig = px.histogram(
        data,
        x=col,
        title=f"Distribution of {col}",
        nbins=20
    )
    
    return fig