"""
LangChain-compatible tools for use with create_react_agent.
"""
import boto3
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from langchain.tools import Tool
from agents_playground.logger import agent_logger
import os


def _create_cost_plot(service_costs: list) -> Optional[dict]:
    """
    Create a minimal bar plot data structure for AWS service costs.
    
    Args:
        service_costs: List of dictionaries with service cost data
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if not service_costs:
        return None
    
    try:
        # Convert to DataFrame for processing
        df = pd.DataFrame(service_costs)
        
        # Take top 10 services for better visualization
        top_services = df.head(10)
        
        # Create minimal plot data structure instead of full Plotly figure
        plot_data = {
            "data": [{
                "type": "bar",
                "x": top_services['service'].tolist(),
                "y": top_services['cost'].tolist(),
                "marker": {
                    "color": "steelblue"
                }
            }],
            "layout": {
                "title": "AWS Service Costs - Last Month (Top 10)",
                "xaxis": {
                    "title": "AWS Service",
                    "tickangle": -45
                },
                "yaxis": {
                    "title": "Cost (USD)",
                    "tickformat": "$,.2f"
                },
                "height": 500,
                "showlegend": False
            }
        }
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "cost_plot_creation"})
        return None


def _get_last_month_costs(query: str = "") -> str:
    """
    Fetch AWS costs for the last month using Cost Explorer API and generate visualization.
    
    Args:
        query: User query (not used in implementation, but required by LangChain Tool interface)
    
    Returns:
        Formatted string with cost information and plot data
    """
    try:
        # Initialize Cost Explorer client
        ce_client = boto3.client('ce')
        
        # Calculate date range for last month
        today = datetime.now().date()
        first_day_current_month = today.replace(day=1)
        last_day_previous_month = first_day_current_month - timedelta(days=1)
        first_day_previous_month = last_day_previous_month.replace(day=1)
        
        start_date = first_day_previous_month.strftime('%Y-%m-%d')
        end_date = first_day_current_month.strftime('%Y-%m-%d')
        
        # Query Cost Explorer for cost and usage data
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        # Process the response
        cost_data = response.get('ResultsByTime', [])
        if not cost_data:
            return "No cost data available for the last month."
        
        total_cost = 0
        service_costs = []
        
        for result_data in cost_data:
            period = result_data.get('TimePeriod', {})
            groups = result_data.get('Groups', [])
            
            for group in groups:
                service_name = group.get('Keys', ['Unknown'])[0]
                metrics = group.get('Metrics', {})
                blended_cost = metrics.get('BlendedCost', {})
                amount = float(blended_cost.get('Amount', '0'))
                currency = blended_cost.get('Unit', 'USD')
                
                if amount > 0:
                    total_cost += amount
                    service_costs.append({
                        'service': service_name,
                        'cost': amount,
                        'currency': currency
                    })
        
        # Sort services by cost (descending)
        service_costs.sort(key=lambda x: x['cost'], reverse=True)
        
        # Generate cost visualization plot
        plot = _create_cost_plot(service_costs)
        plot_json = ""
        plot_info = ""
        
        if plot:
            # Serialize minimal plot data to JSON for transmission
            # Use compact JSON serialization to minimize size
            compact_json = json.dumps(plot, separators=(',', ':'))
            plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            plot_info = "\n\n📊 A cost breakdown chart has been generated and will be displayed below the text report."
        
        # Format the result
        result_lines = [
            f"AWS Cost Report for {start_date} to {end_date}",
            f"Total Cost: ${total_cost:.2f} USD",
            "",
            "Top Services by Cost:"
        ]
        
        for service in service_costs[:10]:  # Show top 10 services
            result_lines.append(
                f"  • {service['service']}: ${service['cost']:.2f} {service['currency']}"
            )
        
        if len(service_costs) > 10:
            result_lines.append(f"  ... and {len(service_costs) - 10} more services")
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info("Successfully fetched AWS last month cost data with plot")
        return result
        
    except Exception as e:
        error_msg = f"Error fetching AWS last month costs: {str(e)}"
        agent_logger.log_error(e, {"context": "aws_last_month_cost_tool"})
        return error_msg


def _get_current_month_costs(query: str = "") -> str:
    """
    Fetch AWS costs for the current month using Cost Explorer API.
    
    Args:
        query: User query (not used in implementation, but required by LangChain Tool interface)
    
    Returns:
        Formatted string with cost information
    """
    try:
        # Initialize Cost Explorer client
        ce_client = boto3.client('ce')
        
        # Calculate date range for current month
        today = datetime.now().date()
        first_day_current_month = today.replace(day=1)
        
        start_date = first_day_current_month.strftime('%Y-%m-%d')
        end_date = today.strftime('%Y-%m-%d')
        
        # Query Cost Explorer for cost and usage data
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost', 'UnblendedCost', 'UsageQuantity'],
            GroupBy=[
                {
                    'Type': 'DIMENSION',
                    'Key': 'SERVICE'
                }
            ]
        )
        
        # Process the response
        cost_data = response.get('ResultsByTime', [])
        if not cost_data:
            return "No cost data available for the current month."
        
        total_cost = 0
        service_costs = []
        
        for result_data in cost_data:
            period = result_data.get('TimePeriod', {})
            groups = result_data.get('Groups', [])
            
            for group in groups:
                service_name = group.get('Keys', ['Unknown'])[0]
                metrics = group.get('Metrics', {})
                blended_cost = metrics.get('BlendedCost', {})
                amount = float(blended_cost.get('Amount', '0'))
                currency = blended_cost.get('Unit', 'USD')
                
                if amount > 0:
                    total_cost += amount
                    service_costs.append({
                        'service': service_name,
                        'cost': amount,
                        'currency': currency
                    })
        
        # Sort services by cost (descending)
        service_costs.sort(key=lambda x: x['cost'], reverse=True)
        
        # Generate cost visualization plot
        plot = _create_cost_plot(service_costs)
        plot_json = ""
        plot_info = ""
        
        if plot:
            # Serialize minimal plot data to JSON for transmission
            # Use compact JSON serialization to minimize size
            compact_json = json.dumps(plot, separators=(',', ':'))
            plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            plot_info = "\n\n📊 A cost breakdown chart has been generated and will be displayed below the text report."
        
        # Format the result
        result_lines = [
            f"AWS Cost Report for {start_date} to {end_date} (Current Month)",
            f"Total Cost: ${total_cost:.2f} USD",
            "",
            "Top Services by Cost:"
        ]
        
        for service in service_costs[:10]:  # Show top 10 services
            result_lines.append(
                f"  • {service['service']}: ${service['cost']:.2f} {service['currency']}"
            )
        
        if len(service_costs) > 10:
            result_lines.append(f"  ... and {len(service_costs) - 10} more services")
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info("Successfully fetched AWS current month cost data with plot")
        return result
        
    except Exception as e:
        error_msg = f"Error fetching AWS current month costs: {str(e)}"
        agent_logger.log_error(e, {"context": "aws_current_month_cost_tool"})
        return error_msg


def _create_usage_plot(usage_data: pd.DataFrame, metric: str) -> Optional[dict]:
    """
    Create a minimal line plot data structure for usage metrics over time.
    
    Args:
        usage_data: DataFrame with usage data
        metric: The metric column to plot
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if usage_data.empty or metric not in usage_data.columns:
        return None
    
    try:
        # Create minimal plot data structure for time series
        # Convert datetime to strings to avoid JSON serialization issues
        x_values = usage_data['day'].dt.strftime('%Y-%m-%d').tolist() if 'day' in usage_data.columns else list(range(len(usage_data)))
        y_values = usage_data[metric].tolist()
        
        plot_data = {
            "data": [{
                "type": "scatter",
                "mode": "lines+markers",
                "x": x_values,
                "y": y_values,
                "name": metric,
                "line": {
                    "color": "steelblue"
                },
                "marker": {
                    "color": "steelblue",
                    "size": 6
                }
            }],
            "layout": {
                "title": f"Product Usage: {metric} Over Time",
                "xaxis": {
                    "title": "Date",
                    "type": "date"
                },
                "yaxis": {
                    "title": metric,
                    "tickformat": ",.0f"
                },
                "height": 500,
                "showlegend": False
            }
        }
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "usage_plot_creation"})
        return None


def _get_usage_report(query: str = "") -> str:
    """
    Analyze product usage data from CSV reports in the resources/data directory.
    This function uses session-stored cluster information or extracts it from query.
    Provides lean responses for specific metric queries.
    
    Args:
        query: User query that may contain cluster information and specific metric requests
    
    Returns:
        Formatted string with usage analysis and plot data, or lean response for specific metrics
    """
    # Get cluster from session state or extract from query
    cluster = ""
    report_name = "usage_report.csv"
    
    try:
        # Try to get session cluster functions from app module
        import streamlit as st
        if hasattr(st, 'session_state'):
            # Check if there's a stored cluster in session
            stored_cluster = getattr(st.session_state, 'current_cluster', None)
            cluster_asked = getattr(st.session_state, 'cluster_asked', False)
            
            if stored_cluster:
                cluster = stored_cluster
            elif not cluster_asked:
                # Look for cluster in current query
                import re
                cluster_patterns = [
                    r'for\s+([a-zA-Z0-9-]+(?:-[a-zA-Z0-9-]+)*)',  # "for cluster-name"
                    r'cluster\s+([a-zA-Z0-9-]+(?:-[a-zA-Z0-9-]+)*)',  # "cluster cluster-name"
                    r'([a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9-]+)',  # pattern like "dev-mt-eks-core"
                ]
                
                for pattern in cluster_patterns:
                    match = re.search(pattern, query, re.IGNORECASE)
                    if match:
                        cluster = match.group(1)
                        # Store cluster in session for future use
                        st.session_state.current_cluster = cluster
                        st.session_state.cluster_asked = True
                        break
                
                # If no cluster found in query and haven't asked yet, ask for it
                if not cluster:
                    st.session_state.cluster_asked = True
                    return "Which cluster are you asking about?"
    except:
        # Fallback to query extraction if session state is not available
        import re
        cluster_patterns = [
            r'for\s+([a-zA-Z0-9-]+(?:-[a-zA-Z0-9-]+)*)',  # "for cluster-name"  
            r'cluster\s+([a-zA-Z0-9-]+(?:-[a-zA-Z0-9-]+)*)',  # "cluster cluster-name"
            r'([a-zA-Z0-9]+-[a-zA-Z0-9]+-[a-zA-Z0-9-]+)',  # pattern like "dev-mt-eks-core"
        ]
        
        for pattern in cluster_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                cluster = match.group(1)
                break
    
    # Check if this is a specific metric query (lean response needed)
    query_lower = query.lower()
    specific_metric_patterns = {
        'sessionCount': [r'\bsessioncount\b', r'\bnumber of sessions?\b', r'\btotal sessions?\b', r'\bsession count\b'],
        'apiCallCount': [r'\bapicallcount\b', r'\bnumber of (api )?calls?\b', r'\btotal (api )?calls?\b', r'\bapi call count\b'],
        'viewCount': [r'\bviewcount\b', r'\bnumber of views?\b', r'\btotal views?\b', r'\bview count\b'],
        'customEventCount': [r'\bcustomeventcount\b', r'\bnumber of events?\b', r'\btotal events?\b', r'\bevent count\b'],
        'playableSessionCount': [r'\bplayablesessioncount\b', r'\bplayable sessions?\b', r'\bplayable session count\b']
    }
    
    requested_metric = None
    for metric, patterns in specific_metric_patterns.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                requested_metric = metric
                break
        if requested_metric:
            break
    try:
        # Construct path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        csv_path = os.path.join(project_root, "resources", "data", report_name)
        
        if not os.path.exists(csv_path):
            return f"Usage report file not found: {report_name}"
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        
        if df.empty:
            return f"No data found in {report_name}"
        
        # Filter by cluster if specified
        if cluster and 'clusterName' in df.columns:
            df = df[df['clusterName'] == cluster]
            if df.empty:
                return f"No data found for cluster '{cluster}' in {report_name}"
        
        # Convert day column to datetime if it exists
        if 'day' in df.columns:
            df['day'] = pd.to_datetime(df['day'], dayfirst=True)
        
        # Generate summary statistics
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
        total_days = len(df)
        date_range = f"{df['day'].min().strftime('%Y-%m-%d')} to {df['day'].max().strftime('%Y-%m-%d')}" if 'day' in df.columns else "Unknown"
        
        # Find the most significant metric (highest average)
        main_metrics = ['apiCallCount', 'sessionCount', 'viewCount']
        available_metrics = [col for col in main_metrics if col in df.columns]
        
        if available_metrics:
            metric_averages = {col: df[col].mean() for col in available_metrics}
            primary_metric = max(metric_averages, key=metric_averages.get)
        else:
            primary_metric = numeric_columns[0] if len(numeric_columns) > 0 else None
        
        # Handle specific metric requests (lean responses)
        if requested_metric and requested_metric in df.columns:
            total_value = df[requested_metric].sum()
            
            # Check if user also wants a plot (keywords: plot, chart, graph, visualize)
            wants_plot = any(keyword in query_lower for keyword in ['plot', 'chart', 'graph', 'visualiz', 'trend'])
            
            # Generate plot if requested
            plot_json = ""
            if wants_plot and 'day' in df.columns:
                plot = _create_usage_plot(df, requested_metric)
                if plot:
                    compact_json = json.dumps(plot, separators=(',', ':'))
                    plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            
            # Return lean response with just the requested metric
            lean_response = f"{total_value:,}"
            if wants_plot and plot_json:
                lean_response += plot_json
            
            agent_logger.info(f"Provided lean response for {requested_metric}: {total_value:,}")
            return lean_response
        
        # Generate usage visualization plot
        plot = None
        plot_json = ""
        plot_info = ""
        
        if primary_metric and 'day' in df.columns:
            plot = _create_usage_plot(df, primary_metric)
            if plot:
                # Serialize minimal plot data to JSON for transmission
                compact_json = json.dumps(plot, separators=(',', ':'))
                plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
                plot_info = f"\n\n📊 A usage trend chart for {primary_metric} has been generated and will be displayed below the report."
        
        # Format the result
        result_lines = [
            f"Product Usage Report Analysis",
            f"Report: {report_name}",
            f"Cluster: {cluster if cluster else 'All clusters'}",
            f"Period: {date_range}",
            f"Total Days: {total_days}",
            "",
            "Key Metrics Summary:"
        ]
        
        # Add summary statistics for main metrics
        for col in available_metrics:
            total = df[col].sum()
            avg = df[col].mean()
            max_val = df[col].max()
            result_lines.extend([
                f"  • {col}:",
                f"    - Total: {total:,}",
                f"    - Daily Average: {avg:,.1f}",
                f"    - Peak Day: {max_val:,}"
            ])
        
        # Add other numeric metrics if available
        other_metrics = [col for col in numeric_columns if col not in available_metrics]
        if other_metrics:
            result_lines.append("\nOther Metrics:")
            for col in other_metrics[:5]:  # Limit to top 5 other metrics
                total = df[col].sum()
                avg = df[col].mean()
                result_lines.append(f"  • {col}: Total {total:,}, Avg {avg:,.1f}")
        
        # Add insights
        if len(df) > 1 and primary_metric:
            latest_value = df[primary_metric].iloc[-1]
            previous_value = df[primary_metric].iloc[-2] if len(df) > 1 else latest_value
            change_pct = ((latest_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
            
            result_lines.extend([
                "",
                "Recent Trends:",
                f"  • Latest {primary_metric}: {latest_value:,}",
                f"  • Change from previous day: {change_pct:+.1f}%"
            ])
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info(f"Successfully analyzed usage report: {report_name}")
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing usage report {report_name}: {str(e)}"
        agent_logger.log_error(e, {"context": "usage_report_analysis", "report_name": report_name})
        return error_msg


# Create LangChain tools
aws_last_month_costs_tool = Tool(
    name="get_last_month_aws_costs",
    description="Fetch AWS costs for the last month. Use this when the user asks about previous month's AWS spending or costs.",
    func=_get_last_month_costs
)

aws_current_month_costs_tool = Tool(
    name="get_current_month_aws_costs", 
    description="Fetch AWS costs for the current month (month-to-date). Use this when the user asks about current month's AWS spending or costs.",
    func=_get_current_month_costs
)

usage_report_tool = Tool(
    name="get_product_usage_report",
    description="Analyze product usage data from CSV reports. Use this when the user asks about product usage, user engagement, API calls, sessions, views, or product analytics. Note: User should specify which tenant/cluster they are asking about (e.g., 'dev-mt-eks-core').",
    func=_get_usage_report
)

# Export available tools
available_tools = [
    aws_last_month_costs_tool,
    aws_current_month_costs_tool,
    usage_report_tool
]