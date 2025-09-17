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


def _create_cluster_resources_plot(df: pd.DataFrame, metric_type: str) -> Optional[dict]:
    """
    Create a plot for cluster resources data visualization.
    
    Args:
        df: DataFrame containing cluster resources data
        metric_type: Type of metric to plot ('usage' or 'cost')
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if df.empty:
        return None
    
    try:
        if metric_type == 'cost':
            # Group by resource type and sum costs
            cost_summary = df.groupby('resource_type')['price_per_day'].sum().sort_values(ascending=False)
            
            plot_data = {
                "data": [{
                    "type": "bar",
                    "x": cost_summary.index.tolist(),
                    "y": cost_summary.values.tolist(),
                    "marker": {
                        "color": "steelblue"
                    }
                }],
                "layout": {
                    "title": "Cluster Resources Cost by Resource Type",
                    "xaxis": {
                        "title": "Resource Type",
                        "tickangle": -45
                    },
                    "yaxis": {
                        "title": "Cost (USD/day)",
                        "tickformat": "$,.2f"
                    },
                    "height": 500,
                    "showlegend": False
                }
            }
        else:  # usage
            # For usage metrics, focus on CPU utilization over time
            cpu_data = df[df['metric_name'] == 'CPUUtilization'].copy()
            if not cpu_data.empty:
                cpu_data['timestamp'] = pd.to_datetime(cpu_data['timestamp'])
                cpu_summary = cpu_data.groupby(cpu_data['timestamp'].dt.date)['value'].mean()
                
                plot_data = {
                    "data": [{
                        "type": "scatter",
                        "mode": "lines+markers",
                        "x": [str(date) for date in cpu_summary.index],
                        "y": cpu_summary.values.tolist(),
                        "marker": {
                            "color": "green"
                        },
                        "line": {
                            "color": "green"
                        }
                    }],
                    "layout": {
                        "title": "Average CPU Utilization Over Time",
                        "xaxis": {
                            "title": "Date"
                        },
                        "yaxis": {
                            "title": "CPU Utilization (%)",
                            "tickformat": ".1f"
                        },
                        "height": 500,
                        "showlegend": False
                    }
                }
            else:
                return None
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "cluster_resources_plot_creation"})
        return None


def _create_generic_dimension_plot(df: pd.DataFrame, dimension: str, value_column: str = 'value') -> Optional[dict]:
    """
    Create a generic plot for any dimension aggregated daily.
    
    Args:
        df: DataFrame containing the data
        dimension: Column name to aggregate by (e.g., 'resource_type', 'instance_type')
        value_column: Column name containing values to aggregate
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if df.empty:
        return None
    
    try:
        # Convert timestamp to date for daily aggregation
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        # Check if it's a categorical dimension or numeric aggregation
        if dimension in df.columns and not pd.api.types.is_numeric_dtype(df[dimension]):
            # Categorical dimension - aggregate by dimension over time
            daily_agg = df.groupby(['date', dimension])[value_column].sum().reset_index()
            
            # Create a line plot for each category
            plot_data = {
                "data": [],
                "layout": {
                    "title": f"Daily {dimension.title().replace('_', ' ')} Trends",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": f"Total {value_column.title().replace('_', ' ')}"},
                    "height": 500
                }
            }
            
            # Add trace for each unique dimension value (limit to top 10)
            top_dimensions = daily_agg.groupby(dimension)[value_column].sum().nlargest(10).index
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            
            for i, dim_value in enumerate(top_dimensions):
                dim_data = daily_agg[daily_agg[dimension] == dim_value]
                plot_data["data"].append({
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": [str(date) for date in dim_data['date']],
                    "y": dim_data[value_column].tolist(),
                    "name": str(dim_value),
                    "marker": {"color": colors[i % len(colors)]},
                    "line": {"color": colors[i % len(colors)]}
                })
        else:
            # Numeric dimension - aggregate daily totals
            daily_totals = df.groupby('date')[value_column].sum().reset_index()
            
            plot_data = {
                "data": [{
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": [str(date) for date in daily_totals['date']],
                    "y": daily_totals[value_column].tolist(),
                    "marker": {"color": "steelblue"},
                    "line": {"color": "steelblue"}
                }],
                "layout": {
                    "title": f"Daily {value_column.title().replace('_', ' ')} Trends",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": f"Total {value_column.title().replace('_', ' ')}"},
                    "height": 500,
                    "showlegend": False
                }
            }
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "generic_dimension_plot_creation", "dimension": dimension})
        return None


def _get_cluster_dimension_analytics(query: str = "") -> str:
    """
    Generic tool that reads the CSV file, aggregates daily data for any specified dimension(s) and plots it.
    Supports queries like 'plot resource_type daily', 'show instance_type trends', 'aggregate price_per_day by resource_type'.
    
    Args:
        query: User query string specifying the dimension(s) to analyze
    
    Returns:
        Formatted string with dimension analytics and plot data
    """
    try:
        # Path to the CSV file
        csv_path = "resources/data/daily_resources_utilizations.csv"
        
        if not os.path.exists(csv_path):
            return f"Error: CSV file not found at {csv_path}"
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        agent_logger.info(f"Loaded cluster resources data with {len(df)} records")
        
        if df.empty:
            return "No data found in the cluster resources file."
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', utc=True)
        
        # Available dimensions and value columns
        available_dimensions = ['clusterName', 'region', 'cluster_version', 'instance_type', 
                               'metric_name', 'resource_id', 'resource_type', 'volume_id', 'volume_type', 'unit']
        available_values = ['price_per_day', 'size_gb', 'value']
        
        # Parse query to identify dimension and value column
        query_lower = query.lower()
        
        # Find dimension in query
        dimension = None
        for dim in available_dimensions:
            if dim.lower() in query_lower or dim.lower().replace('_', ' ') in query_lower:
                dimension = dim
                break
        
        # Find value column in query (default to 'value')
        value_column = 'value'  # default
        for val_col in available_values:
            if val_col.lower() in query_lower or val_col.lower().replace('_', ' ') in query_lower:
                value_column = val_col
                break
        
        # If no specific dimension found, try to infer from common patterns
        if not dimension:
            if any(word in query_lower for word in ['instance', 'type', 'ec2']):
                dimension = 'instance_type'
            elif any(word in query_lower for word in ['resource', 'resources']):
                dimension = 'resource_type'
            elif any(word in query_lower for word in ['metric', 'metrics']):
                dimension = 'metric_name'
            elif any(word in query_lower for word in ['cluster', 'clusters']):
                dimension = 'clusterName'
            else:
                dimension = 'resource_type'  # default fallback
        
        # If asking about cost/price, use price_per_day as value column
        if any(word in query_lower for word in ['cost', 'price', 'pricing', 'expense', 'spend']):
            value_column = 'price_per_day'
        
        # Filter data for the specified dimension and value column
        filtered_df = df.dropna(subset=[dimension, value_column])
        
        if filtered_df.empty:
            return f"No data found for dimension '{dimension}' with value column '{value_column}'."
        
        # Create date column for daily aggregation
        filtered_df['date'] = filtered_df['timestamp'].dt.date
        
        # Aggregate data daily
        if dimension in ['clusterName', 'region', 'instance_type', 'metric_name', 'resource_type', 'volume_type', 'unit']:
            # Categorical dimension - group by date and dimension
            daily_agg = filtered_df.groupby(['date', dimension])[value_column].sum().reset_index()
            
            # Get summary statistics
            total_value = filtered_df[value_column].sum()
            unique_dimensions = filtered_df[dimension].nunique()
            date_range = f"{filtered_df['date'].min()} to {filtered_df['date'].max()}"
            
            result_lines = [
                f"Daily {dimension.title().replace('_', ' ')} Analytics",
                f"Period: {date_range}",
                f"Total {value_column.replace('_', ' ').title()}: {total_value:,.2f}",
                f"Unique {dimension.replace('_', ' ').title()} Values: {unique_dimensions}",
                "",
                f"Top {dimension.replace('_', ' ').title()} by Total {value_column.replace('_', ' ').title()}:"
            ]
            
            # Show top values
            top_values = filtered_df.groupby(dimension)[value_column].sum().sort_values(ascending=False).head(10)
            for dim_val, total_val in top_values.items():
                percentage = (total_val / total_value) * 100 if total_value > 0 else 0
                result_lines.append(f"  • {dim_val}: {total_val:,.2f} ({percentage:.1f}%)")
            
        else:
            # Numeric aggregation - just sum by date
            daily_totals = filtered_df.groupby('date')[value_column].sum().reset_index()
            total_value = filtered_df[value_column].sum()
            avg_daily = daily_totals[value_column].mean()
            date_range = f"{daily_totals['date'].min()} to {daily_totals['date'].max()}"
            
            result_lines = [
                f"Daily {value_column.replace('_', ' ').title()} Analytics",
                f"Period: {date_range}",
                f"Total {value_column.replace('_', ' ').title()}: {total_value:,.2f}",
                f"Average Daily {value_column.replace('_', ' ').title()}: {avg_daily:,.2f}",
                f"Number of Days: {len(daily_totals)}"
            ]
        
        # Generate plot
        plot = _create_generic_dimension_plot(filtered_df, dimension, value_column)
        plot_json = ""
        plot_info = ""
        
        if plot:
            compact_json = json.dumps(plot, separators=(',', ':'))
            plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            plot_info = f"\n\n📊 A daily trends chart for {dimension.replace('_', ' ')} has been generated and will be displayed below."
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info(f"Successfully analyzed dimension '{dimension}' with value '{value_column}'")
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing cluster dimension: {str(e)}"
        agent_logger.log_error(e, {"context": "cluster_dimension_analytics"})
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

cluster_dimension_analytics_tool = Tool(
    name="get_cluster_dimension_analytics",
    description="Generic tool that reads the cluster resources CSV file, aggregates daily data for any specified dimension(s) and plots it. Use this when the user asks to analyze, aggregate, or plot any dimension from the cluster data (e.g., 'plot resource_type daily', 'show instance_type trends', 'aggregate price_per_day by resource_type', 'analyze metric_name', 'plot size_gb by volume_type'). Supports all available dimensions: clusterName, region, instance_type, metric_name, resource_type, volume_type, etc.",
    func=_get_cluster_dimension_analytics
)

# Export available tools
available_tools = [
    aws_last_month_costs_tool,
    aws_current_month_costs_tool,
    usage_report_tool,
    cluster_dimension_analytics_tool
]