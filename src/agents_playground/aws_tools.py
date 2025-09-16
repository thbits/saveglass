"""
AWS integration tools for the agents playground.
"""
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from agents_playground.logger import agent_logger
from agents_playground.models import AgentState


def get_last_month_costs(state: AgentState, **kwargs) -> AgentState:
    """
    Tool to fetch AWS costs for the last month using Cost Explorer API.
    
    Args:
        state: Current agent state
        **kwargs: Additional parameters (not used in this implementation)
    
    Returns:
        Updated agent state with cost information
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
            result = "No cost data available for the last month."
        else:
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
            
            result = "\n".join(result_lines)
        
        # Add result to agent state
        if "tool_results" not in state.metadata:
            state.metadata["tool_results"] = []
        
        state.metadata["tool_results"].append(f"AWS Cost Tool: {result}")
        
        agent_logger.info("Successfully fetched AWS cost data")
        
    except Exception as e:
        error_msg = f"Error fetching AWS costs: {str(e)}"
        
        if "tool_results" not in state.metadata:
            state.metadata["tool_results"] = []
        
        state.metadata["tool_results"].append(f"AWS Cost Tool Error: {error_msg}")
        
        agent_logger.log_error(e, {"context": "aws_cost_tool"})
    
    return state


def get_current_month_costs(state: AgentState, **kwargs) -> AgentState:
    """
    Tool to fetch AWS costs for the current month (month-to-date).
    
    Args:
        state: Current agent state
        **kwargs: Additional parameters
    
    Returns:
        Updated agent state with cost information
    """
    try:
        # Initialize Cost Explorer client
        ce_client = boto3.client('ce')
        
        # Calculate date range for current month
        today = datetime.now().date()
        first_day_current_month = today.replace(day=1)
        
        start_date = first_day_current_month.strftime('%Y-%m-%d')
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')  # Tomorrow for inclusive range
        
        # Query Cost Explorer
        response = ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='MONTHLY',
            Metrics=['BlendedCost'],
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
            result = "No cost data available for the current month."
        else:
            total_cost = 0
            service_costs = []
            
            for result_data in cost_data:
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
            
            # Format the result
            result_lines = [
                f"AWS Cost Report for Current Month ({start_date} to {today})",
                f"Month-to-Date Total: ${total_cost:.2f} USD",
                "",
                "Top Services by Cost:"
            ]
            
            for service in service_costs[:10]:
                result_lines.append(
                    f"  • {service['service']}: ${service['cost']:.2f} {service['currency']}"
                )
            
            if len(service_costs) > 10:
                result_lines.append(f"  ... and {len(service_costs) - 10} more services")
            
            result = "\n".join(result_lines)
        
        # Add result to agent state
        if "tool_results" not in state.metadata:
            state.metadata["tool_results"] = []
        
        state.metadata["tool_results"].append(f"AWS Current Month Cost Tool: {result}")
        
        agent_logger.info("Successfully fetched current month AWS cost data")
        
    except Exception as e:
        error_msg = f"Error fetching current month AWS costs: {str(e)}"
        
        if "tool_results" not in state.metadata:
            state.metadata["tool_results"] = []
        
        state.metadata["tool_results"].append(f"AWS Current Month Cost Tool Error: {error_msg}")
        
        agent_logger.log_error(e, {"context": "aws_current_month_cost_tool"})
    
    return state


# def get_cost_forecast(state: AgentState, **kwargs) -> AgentState:
#     """
#     Tool to get AWS cost forecast for the next month.
#
#     Args:
#         state: Current agent state
#         **kwargs: Additional parameters
#
#     Returns:
#         Updated agent state with forecast information
#     """
#     try:
#         # Initialize Cost Explorer client
#         ce_client = boto3.client('ce')
#
#         # Calculate date range for forecast (next month)
#         today = datetime.now().date()
#         next_month_start = (today.replace(day=28) + timedelta(days=4)).replace(day=1)
#         next_month_end = (next_month_start + timedelta(days=32)).replace(day=1)
#
#         start_date = next_month_start.strftime('%Y-%m-%d')
#         end_date = next_month_end.strftime('%Y-%m-%d')
#
#         # Query Cost Explorer for forecast
#         response = ce_client.get_cost_forecast(
#             TimePeriod={
#                 'Start': start_date,
#                 'End': end_date
#             },
#             Metric='BLENDED_COST',
#             Granularity='MONTHLY'
#         )
#
#         # Process the response
#         total = response.get('Total', {})
#         forecasted_amount = float(total.get('Amount', '0'))
#         currency = total.get('Unit', 'USD')
#
#         result = f"AWS Cost Forecast for {start_date} to {end_date}: ${forecasted_amount:.2f} {currency}"
#
#         # Add result to agent state
#         if "tool_results" not in state.metadata:
#             state.metadata["tool_results"] = []
#
#         state.metadata["tool_results"].append(f"AWS Cost Forecast Tool: {result}")
#
#         agent_logger.info("Successfully fetched AWS cost forecast")
#
#     except Exception as e:
#         error_msg = f"Error fetching AWS cost forecast: {str(e)}"
#
#         if "tool_results" not in state.metadata:
#             state.metadata["tool_results"] = []
#
#         state.metadata["tool_results"].append(f"AWS Cost Forecast Tool Error: {error_msg}")
#
#         agent_logger.log_error(e, {"context": "aws_cost_forecast_tool"})
#
#     return state