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


def _create_eks_clusters_plot(df: pd.DataFrame, plot_type: str = "cost") -> Optional[dict]:
    """
    Create a plot for EKS clusters data visualization.
    
    Args:
        df: DataFrame containing EKS clusters data
        plot_type: Type of plot ('cost', 'timeline', or 'comparison')
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if df.empty:
        return None
    
    try:
        if plot_type == "cost":
            # Group by cluster and sum daily costs
            cost_summary = df.groupby('cluster_name')['price_per_day'].sum().sort_values(ascending=False)
            
            plot_data = {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in cost_summary.index.tolist()],
                    "y": [float(y) for y in cost_summary.values.tolist()],
                    "marker": {
                        "color": "steelblue"
                    }
                }],
                "layout": {
                    "title": "EKS Cluster Costs (Total)",
                    "xaxis": {
                        "title": "Cluster Name",
                        "tickangle": -45
                    },
                    "yaxis": {
                        "title": "Total Cost (USD/day)",
                        "tickformat": "$,.2f"
                    },
                    "height": 500,
                    "showlegend": False
                }
            }
        elif plot_type == "timeline":
            # Timeline of costs over time
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            timeline_data = df.groupby(['date', 'cluster_name'])['price_per_day'].sum().reset_index()
            
            plot_data = {
                "data": [],
                "layout": {
                    "title": "EKS Cluster Costs Over Time",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Daily Cost (USD)", "tickformat": "$,.2f"},
                    "height": 500
                }
            }
            
            # Add trace for each cluster
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            for i, cluster in enumerate(timeline_data['cluster_name'].unique()):
                cluster_data = timeline_data[timeline_data['cluster_name'] == cluster]
                plot_data["data"].append({
                    "type": "scatter",
                    "mode": "lines+markers",
                    "x": [str(date) for date in cluster_data['date']],
                    "y": [float(y) for y in cluster_data['price_per_day'].tolist()],
                    "name": str(cluster),
                    "marker": {"color": colors[i % len(colors)]},
                    "line": {"color": colors[i % len(colors)]}
                })
        else:  # comparison
            # Compare clusters by region and version
            comparison_data = df.groupby(['cluster_region', 'cluster_version'])['price_per_day'].sum().reset_index()
            
            plot_data = {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": [str(x) for x in comparison_data['cluster_region'].tolist()],
                    "y": [str(y) for y in comparison_data['cluster_version'].tolist()],
                    "marker": {
                        "size": [float(s) for s in comparison_data['price_per_day'].tolist()],
                        "color": [float(c) for c in comparison_data['price_per_day'].tolist()],
                        "colorscale": "Viridis",
                        "showscale": True,
                        "colorbar": {"title": "Cost (USD/day)"}
                    },
                    "text": [f"Cost: ${float(cost):.2f}/day" for cost in comparison_data['price_per_day']],
                    "hovertemplate": "Region: %{x}<br>Version: %{y}<br>%{text}<extra></extra>"
                }],
                "layout": {
                    "title": "EKS Clusters: Region vs Version (Size = Cost)",
                    "xaxis": {"title": "Region"},
                    "yaxis": {"title": "Kubernetes Version"},
                    "height": 500,
                    "showlegend": False
                }
            }
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "eks_clusters_plot_creation", "plot_type": plot_type})
        return None


def _create_ebs_volumes_plot(df: pd.DataFrame, plot_type: str = "cost") -> Optional[dict]:
    """
    Create a plot for EBS volumes data visualization.
    
    Args:
        df: DataFrame containing EBS volumes data
        plot_type: Type of plot ('cost', 'size', 'type_comparison', or 'cost_per_gb')
    
    Returns:
        Minimal plot data dictionary for efficient transmission
    """
    if df.empty:
        return None
    
    try:
        if plot_type == "cost":
            # Top volumes by cost
            volume_costs = df.groupby('resource_id')['price_per_day'].sum().sort_values(ascending=False).head(15)
            
            plot_data = {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in volume_costs.index.tolist()],
                    "y": [float(y) for y in volume_costs.values.tolist()],
                    "marker": {
                        "color": "steelblue"
                    }
                }],
                "layout": {
                    "title": "Top 15 EBS Volumes by Daily Cost",
                    "xaxis": {
                        "title": "Volume ID",
                        "tickangle": -45
                    },
                    "yaxis": {
                        "title": "Daily Cost (USD)",
                        "tickformat": "$,.3f"
                    },
                    "height": 500,
                    "showlegend": False
                }
            }
        elif plot_type == "size":
            # Volume size distribution
            size_bins = [0, 50, 100, 200, 500, 1000, float('inf')]
            size_labels = ['0-50GB', '50-100GB', '100-200GB', '200-500GB', '500GB-1TB', '1TB+']
            df['size_category'] = pd.cut(df['size_gb'], bins=size_bins, labels=size_labels, right=False)
            size_distribution = df['size_category'].value_counts().sort_index()
            
            plot_data = {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in size_distribution.index.tolist()],
                    "y": [int(y) for y in size_distribution.values.tolist()],
                    "marker": {
                        "color": "green"
                    }
                }],
                "layout": {
                    "title": "EBS Volume Size Distribution",
                    "xaxis": {
                        "title": "Size Category"
                    },
                    "yaxis": {
                        "title": "Number of Volumes"
                    },
                    "height": 500,
                    "showlegend": False
                }
            }
        elif plot_type == "type_comparison":
            # Compare volume types by cost and count
            type_summary = df.groupby('volume_type').agg({
                'price_per_day': 'sum',
                'resource_id': 'nunique'
            }).reset_index()
            
            plot_data = {
                "data": [{
                    "type": "scatter",
                    "mode": "markers",
                    "x": [int(x) for x in type_summary['resource_id'].tolist()],
                    "y": [float(y) for y in type_summary['price_per_day'].tolist()],
                    "marker": {
                        "size": [float(s) * 20 for s in type_summary['price_per_day'].tolist()],
                        "color": ['blue', 'red', 'green', 'orange', 'purple'][:len(type_summary)],
                        "opacity": 0.7
                    },
                    "text": [str(t) for t in type_summary['volume_type'].tolist()],
                    "hovertemplate": "Type: %{text}<br>Volume Count: %{x}<br>Total Cost: $%{y:.2f}<extra></extra>"
                }],
                "layout": {
                    "title": "EBS Volume Types: Count vs Total Cost",
                    "xaxis": {"title": "Number of Volumes"},
                    "yaxis": {"title": "Total Daily Cost (USD)", "tickformat": "$,.2f"},
                    "height": 500,
                    "showlegend": False
                }
            }
        else:  # cost_per_gb
            # Cost per GB analysis by volume type
            df['cost_per_gb'] = df['price_per_day'] / df['size_gb']
            cost_per_gb_by_type = df.groupby('volume_type')['cost_per_gb'].mean().sort_values(ascending=False)
            
            plot_data = {
                "data": [{
                    "type": "bar",
                    "x": [str(x) for x in cost_per_gb_by_type.index.tolist()],
                    "y": [float(y) for y in cost_per_gb_by_type.values.tolist()],
                    "marker": {
                        "color": "orange"
                    }
                }],
                "layout": {
                    "title": "Average Cost per GB by Volume Type",
                    "xaxis": {
                        "title": "Volume Type"
                    },
                    "yaxis": {
                        "title": "Cost per GB (USD/day)",
                        "tickformat": "$,.4f"
                    },
                    "height": 500,
                    "showlegend": False
                }
            }
        
        return plot_data
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "ebs_volumes_plot_creation", "plot_type": plot_type})
        return None


def _get_ebs_volumes_analytics(query: str = "") -> str:
    """
    Analyze EBS volumes data from the CSV file. Provides information about volume costs,
    sizes, types, and efficiency metrics. Supports various query types like cost analysis,
    volume comparisons, and size distribution analysis.
    
    Args:
        query: User query string about EBS volumes
    
    Returns:
        Formatted string with EBS volumes analytics and plot data
    """
    try:
        # Construct path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        csv_path = os.path.join(project_root, "resources", "data", "ebs_volumes.csv")
        
        if not os.path.exists(csv_path):
            return f"EBS volumes data file not found: {csv_path}"
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        agent_logger.info(f"Loaded EBS volumes data with {len(df)} records")
        
        if df.empty:
            return "No EBS volumes data found in the file."
        
        # Clean up the data - remove rows where resource_id (volume ID) is NaN or empty
        initial_count = len(df)
        
        # Check if resource_id column exists (this contains the actual volume IDs)
        if 'resource_id' not in df.columns:
            return "Error: 'resource_id' column not found in EBS volumes data."
        
        # Remove NaN values from resource_id (which contains the volume IDs)
        df = df.dropna(subset=['resource_id'])
        
        # Remove empty strings
        if df['resource_id'].dtype == 'object':
            df = df[df['resource_id'].str.strip() != '']
        
        agent_logger.info(f"After cleaning: {len(df)} records (removed {initial_count - len(df)} empty rows)")
        
        if df.empty:
            return f"No valid EBS volumes data found after cleaning. Original count: {initial_count}"
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Calculate cost per GB
        df['cost_per_gb'] = df['price_per_day'] / df['size_gb']
        
        # Analyze query to determine what type of analysis is needed
        query_lower = query.lower()
        
        # Determine plot type based on query
        plot_type = "cost"  # default
        if any(word in query_lower for word in ['size', 'distribution', 'capacity']):
            plot_type = "size"
        elif any(word in query_lower for word in ['type', 'compare', 'comparison', 'gp3', 'sc1', 'st1']):
            plot_type = "type_comparison"
        elif any(word in query_lower for word in ['efficiency', 'cost per gb', 'cost/gb', 'expensive']):
            plot_type = "cost_per_gb"
        
        # Check for specific volume ID in query
        volume_filter = None
        for volume_id in df['resource_id'].unique():
            if str(volume_id).lower() in query_lower:
                volume_filter = str(volume_id)
                break
        
        # Filter data if specific volume requested
        filtered_df = df
        if volume_filter:
            filtered_df = df[df['resource_id'] == volume_filter]
            if filtered_df.empty:
                return f"No data found for volume '{volume_filter}'"
        
        # Generate summary statistics (convert numpy types to Python native types)
        total_volumes = int(filtered_df['resource_id'].nunique())
        total_cost = float(filtered_df['price_per_day'].sum())
        total_storage = float(filtered_df['size_gb'].sum())
        avg_cost_per_volume = total_cost / total_volumes if total_volumes > 0 else 0
        avg_cost_per_gb = float(filtered_df['cost_per_gb'].mean())
        date_range = f"{filtered_df['date'].min()} to {filtered_df['date'].max()}"
        
        # Get unique volume types
        volume_types = sorted([str(vt) for vt in filtered_df['volume_type'].unique()])
        
        # Build result
        result_lines = [
            "EBS Volumes Analytics Report",
            f"Analysis Period: {date_range}",
            f"Total Volumes: {total_volumes:,}",
            f"Total Storage: {total_storage:,.0f} GB ({total_storage/1024:.1f} TB)",
            f"Total Daily Cost: ${total_cost:.2f}",
            f"Average Cost per Volume: ${avg_cost_per_volume:.3f}/day",
            f"Average Cost per GB: ${avg_cost_per_gb:.4f}/day",
            "",
            f"Volume Types: {', '.join(volume_types)}",
            ""
        ]
        
        # Add volume-specific details
        if volume_filter:
            volume_data = filtered_df[filtered_df['resource_id'] == volume_filter].iloc[0]
            result_lines.extend([
                f"Volume Details for '{volume_filter}':",
                f"  • Size: {float(volume_data['size_gb']):.0f} GB",
                f"  • Type: {str(volume_data['volume_type'])}",
                f"  • Daily Cost: ${float(volume_data['price_per_day']):.3f}",
                f"  • Cost per GB: ${float(volume_data['cost_per_gb']):.4f}/day",
                ""
            ])
        else:
            # Volume type breakdown
            result_lines.append("Cost Breakdown by Volume Type:")
            type_summary = filtered_df.groupby('volume_type').agg({
                'price_per_day': 'sum',
                'resource_id': 'nunique',
                'size_gb': 'sum',
                'cost_per_gb': 'mean'
            }).sort_values('price_per_day', ascending=False)
            
            for vol_type, row in type_summary.iterrows():
                cost = float(row['price_per_day'])
                count = int(row['resource_id'])
                storage = float(row['size_gb'])
                cost_per_gb = float(row['cost_per_gb'])
                percentage = (cost / total_cost) * 100 if total_cost > 0 else 0
                result_lines.append(
                    f"  • {str(vol_type)}: {count} volumes, {storage:,.0f} GB, "
                    f"${cost:.2f}/day ({percentage:.1f}%), ${cost_per_gb:.4f}/GB"
                )
            
            result_lines.append("")
            
            # Top expensive volumes
            result_lines.append("Top 5 Most Expensive Volumes:")
            top_volumes = filtered_df.nlargest(5, 'price_per_day')
            for _, vol in top_volumes.iterrows():
                result_lines.append(
                    f"  • {str(vol['resource_id'])}: {float(vol['size_gb']):.0f} GB "
                    f"({str(vol['volume_type'])}), ${float(vol['price_per_day']):.3f}/day"
                )
            
            result_lines.append("")
        
        # Add insights
        if len(filtered_df) > 1:
            # Cost efficiency insights
            most_efficient_type = filtered_df.groupby('volume_type')['cost_per_gb'].mean().idxmin()
            least_efficient_type = filtered_df.groupby('volume_type')['cost_per_gb'].mean().idxmax()
            
            result_lines.extend([
                "Efficiency Insights:",
                f"  • Most cost-efficient type: {str(most_efficient_type)}",
                f"  • Least cost-efficient type: {str(least_efficient_type)}",
                ""
            ])
        
        # Generate plot
        plot = _create_ebs_volumes_plot(filtered_df, plot_type)
        plot_json = ""
        plot_info = ""
        
        if plot:
            compact_json = json.dumps(plot, separators=(',', ':'))
            plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            plot_info = f"\n\n📊 An EBS volumes {plot_type} chart has been generated and will be displayed below."
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info(f"Successfully analyzed EBS volumes data with {plot_type} visualization")
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing EBS volumes: {str(e)}"
        agent_logger.log_error(e, {"context": "ebs_volumes_analytics"})
        return error_msg


def _get_eks_clusters_analytics(query: str = "") -> str:
    """
    Analyze EKS clusters data from the CSV file. Provides information about cluster costs,
    regions, versions, and trends. Supports various query types like cost analysis,
    cluster comparisons, and timeline views.
    
    Args:
        query: User query string about EKS clusters
    
    Returns:
        Formatted string with EKS clusters analytics and plot data
    """
    try:
        # Construct path to the CSV file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        csv_path = os.path.join(project_root, "resources", "data", "eks_clusters.csv")
        
        if not os.path.exists(csv_path):
            return f"EKS clusters data file not found: {csv_path}"
        
        # Read CSV data
        df = pd.read_csv(csv_path)
        agent_logger.info(f"Loaded EKS clusters data with {len(df)} records")
        
        if df.empty:
            return "No EKS clusters data found in the file."
        
        # Clean up the data - remove any empty rows
        df = df.dropna(subset=['cluster_name'])
        
        if df.empty:
            return "No valid EKS clusters data found in the file after cleaning."
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Analyze query to determine what type of analysis is needed
        query_lower = query.lower()
        
        # Determine plot type based on query
        plot_type = "cost"  # default
        if any(word in query_lower for word in ['timeline', 'trend', 'over time', 'history']):
            plot_type = "timeline"
        elif any(word in query_lower for word in ['compare', 'comparison', 'region', 'version']):
            plot_type = "comparison"
        
        # Check for specific cluster name in query
        cluster_filter = None
        for cluster_name in df['cluster_name'].unique():
            if str(cluster_name).lower() in query_lower:
                cluster_filter = str(cluster_name)
                break
        
        # Filter data if specific cluster requested
        filtered_df = df
        if cluster_filter:
            filtered_df = df[df['cluster_name'] == cluster_filter]
            if filtered_df.empty:
                return f"No data found for cluster '{cluster_filter}'"
        
        # Generate summary statistics (convert numpy types to Python native types)
        total_clusters = int(filtered_df['cluster_name'].nunique())
        total_cost = float(filtered_df['price_per_day'].sum())
        avg_daily_cost = float(filtered_df.groupby('cluster_name')['price_per_day'].mean().mean())
        date_range = f"{filtered_df['date'].min()} to {filtered_df['date'].max()}"
        
        # Get unique regions and versions (convert to Python strings)
        regions = sorted([str(r) for r in filtered_df['cluster_region'].unique()])
        versions = sorted([str(v) for v in filtered_df['cluster_version'].unique()])
        
        # Build result
        result_lines = [
            "EKS Clusters Analytics Report",
            f"Analysis Period: {date_range}",
            f"Total Clusters: {total_clusters}",
            f"Total Daily Cost: ${total_cost:.2f}",
            f"Average Daily Cost per Cluster: ${avg_daily_cost:.2f}",
            "",
            f"Regions: {', '.join(regions)}",
            f"Kubernetes Versions: {', '.join(versions)}",
            ""
        ]
        
        # Add cluster-specific details
        if cluster_filter:
            cluster_data = filtered_df[filtered_df['cluster_name'] == cluster_filter].iloc[0]
            result_lines.extend([
                f"Cluster Details for '{cluster_filter}':",
                f"  • Region: {str(cluster_data['cluster_region'])}",
                f"  • Kubernetes Version: {str(cluster_data['cluster_version'])}",
                f"  • Daily Cost: ${float(cluster_data['price_per_day']):.2f}",
                ""
            ])
        else:
            result_lines.append("Cluster Cost Breakdown:")
            cluster_costs = filtered_df.groupby('cluster_name')['price_per_day'].sum().sort_values(ascending=False)
            for cluster_name, cost in cluster_costs.head(10).items():
                percentage = (float(cost) / total_cost) * 100 if total_cost > 0 else 0
                result_lines.append(f"  • {str(cluster_name)}: ${float(cost):.2f}/day ({percentage:.1f}%)")
            
            if len(cluster_costs) > 10:
                result_lines.append(f"  ... and {len(cluster_costs) - 10} more clusters")
            result_lines.append("")
        
        # Add insights
        if len(filtered_df) > 1:
            latest_data = filtered_df[filtered_df['date'] == filtered_df['date'].max()]
            previous_data = filtered_df[filtered_df['date'] == filtered_df['date'].unique()[-2]] if len(filtered_df['date'].unique()) > 1 else latest_data
            
            latest_cost = float(latest_data['price_per_day'].sum())
            previous_cost = float(previous_data['price_per_day'].sum())
            cost_change = latest_cost - previous_cost
            cost_change_pct = (cost_change / previous_cost * 100) if previous_cost != 0 else 0
            
            result_lines.extend([
                "Recent Trends:",
                f"  • Latest Daily Cost: ${latest_cost:.2f}",
                f"  • Cost Change: ${cost_change:+.2f} ({cost_change_pct:+.1f}%)",
                ""
            ])
        
        # Generate plot
        plot = _create_eks_clusters_plot(filtered_df, plot_type)
        plot_json = ""
        plot_info = ""
        
        if plot:
            compact_json = json.dumps(plot, separators=(',', ':'))
            plot_json = f"\n\n[PLOT_DATA]{compact_json}[/PLOT_DATA]"
            plot_info = f"\n\n📊 An EKS clusters {plot_type} chart has been generated and will be displayed below."
        
        result = "\n".join(result_lines) + plot_info + plot_json
        agent_logger.info(f"Successfully analyzed EKS clusters data with {plot_type} visualization")
        return result
        
    except Exception as e:
        error_msg = f"Error analyzing EKS clusters: {str(e)}"
        agent_logger.log_error(e, {"context": "eks_clusters_analytics"})
        return error_msg


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

eks_clusters_analytics_tool = Tool(
    name="get_eks_clusters_analytics",
    description="Analyze EKS clusters data including costs, regions, versions, and trends. Use this when the user asks about EKS clusters, cluster costs, cluster comparisons, or wants to see cluster analytics. Supports queries like 'show EKS cluster costs', 'compare clusters by region', 'EKS cluster timeline', 'cluster cost breakdown', or questions about specific clusters like 'dev-mt-eks-core'.",
    func=_get_eks_clusters_analytics
)

ebs_volumes_analytics_tool = Tool(
    name="get_ebs_volumes_analytics",
    description="Analyze EBS volumes data including costs, sizes, types, and efficiency metrics. Use this when the user asks about EBS volumes, storage costs, volume types (gp3, sc1, st1), volume sizes, cost per GB analysis, or storage efficiency. Supports queries like 'show EBS volume costs', 'compare volume types', 'most expensive volumes', 'volume size distribution', 'cost per GB analysis', or questions about specific volumes.",
    func=_get_ebs_volumes_analytics
)

# Export available tools
available_tools = [
    aws_last_month_costs_tool,
    aws_current_month_costs_tool,
    usage_report_tool,
    cluster_dimension_analytics_tool,
    eks_clusters_analytics_tool,
    ebs_volumes_analytics_tool
]