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

# Export available tools
available_tools = [
    aws_last_month_costs_tool,
    aws_current_month_costs_tool
]