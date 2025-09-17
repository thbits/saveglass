"""
Logging configuration and utilities for AI Agents Playground
"""
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any, Optional
import uuid
from contextlib import contextmanager


class AgentLogger:
    """Centralized logging utility for agent operations with tracing support."""
    
    def __init__(self, name: str = "agents_playground", level: int = logging.INFO):
        """Initialize the logger with structured formatting."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create console handler with structured formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        # Create custom formatter
        formatter = StructuredFormatter()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
        
        self.session_id = str(uuid.uuid4())[:8]
        
    def create_request_context(self, user_input: str, provider: str = None) -> Dict[str, Any]:
        """Create a request context for tracing."""
        return {
            "request_id": str(uuid.uuid4())[:8],
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "user_input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
            "provider": provider,
            "trace_start": datetime.utcnow()
        }
    
    def log_agent_request_start(self, context: Dict[str, Any]):
        """Log the start of an agent request."""
        self.logger.info(
            "Agent request started",
            extra={
                "event_type": "agent_request_start",
                "request_id": context["request_id"],
                "session_id": context["session_id"],
                "user_input": context["user_input"],
                "provider": context["provider"],
                "timestamp": context["timestamp"]
            }
        )
    
    def log_agent_request_end(self, context: Dict[str, Any], response: str, success: bool = True, error: str = None, log_full_response: bool = False):
        """Log the end of an agent request."""
        duration = (datetime.utcnow() - context["trace_start"]).total_seconds()
        
        log_data = {
            "event_type": "agent_request_end",
            "request_id": context["request_id"],
            "session_id": context["session_id"],
            "success": success,
            "duration_seconds": round(duration, 3),
            "response_length": len(response) if response else 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add full response content if requested (for debugging)
        if log_full_response and response:
            log_data["full_response"] = response
            # Also check for plot data markers
            has_plot_data = "[PLOT_DATA]" in response and "[/PLOT_DATA]" in response
            log_data["contains_plot_data"] = has_plot_data
            if has_plot_data:
                import re
                plot_pattern = r'\[PLOT_DATA\](.*?)\[/PLOT_DATA\]'
                plot_matches = re.findall(plot_pattern, response, re.DOTALL)
                log_data["plot_data_count"] = len(plot_matches)
                if plot_matches:
                    log_data["plot_data_size"] = len(plot_matches[0])
        
        if error:
            log_data["error"] = str(error)
            
        if success:
            self.logger.info("Agent request completed successfully", extra=log_data)
        else:
            self.logger.error("Agent request failed", extra=log_data)
    
    def log_agent_initialization(self, provider: str, model: str, config: Dict[str, Any]):
        """Log agent initialization."""
        self.logger.info(
            "Agent initialized",
            extra={
                "event_type": "agent_initialization",
                "session_id": self.session_id,
                "provider": provider,
                "model": model,
                "config": config,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_config_update(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Log configuration updates."""
        self.logger.info(
            "Agent configuration updated",
            extra={
                "event_type": "config_update",
                "session_id": self.session_id,
                "old_config": old_config,
                "new_config": new_config,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log errors with context."""
        log_data = {
            "event_type": "error",
            "session_id": self.session_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if context:
            log_data["context"] = context
            
        self.logger.error("Application error occurred", extra=log_data)
    
    def log_visualization_request(self, prompt: str, chart_type: str, success: bool):
        """Log visualization generation requests."""
        self.logger.info(
            "Visualization request processed",
            extra={
                "event_type": "visualization_request",
                "session_id": self.session_id,
                "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                "chart_type": chart_type,
                "success": success,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    def log_full_llm_response(self, request_id: str, user_input: str, full_response: str, provider: str = None):
        """Log the complete LLM response for debugging purposes."""
        import re
        
        # Analyze response content
        response_analysis = {
            "length": len(full_response),
            "line_count": len(full_response.split('\n')),
            "word_count": len(full_response.split()),
        }
        
        # Check for plot data
        has_plot_data = "[PLOT_DATA]" in full_response and "[/PLOT_DATA]" in full_response
        response_analysis["contains_plot_data"] = has_plot_data
        
        if has_plot_data:
            plot_pattern = r'\[PLOT_DATA\](.*?)\[/PLOT_DATA\]'
            plot_matches = re.findall(plot_pattern, full_response, re.DOTALL)
            response_analysis["plot_data_count"] = len(plot_matches)
            response_analysis["plot_data_sizes"] = [len(match) for match in plot_matches]
            
            # Check if plot data appears to be truncated
            for i, match in enumerate(plot_matches):
                try:
                    import json
                    json.loads(match.strip())
                    response_analysis[f"plot_{i}_valid_json"] = True
                except json.JSONDecodeError:
                    response_analysis[f"plot_{i}_valid_json"] = False
        
        # Check for duplicated content patterns
        lines = full_response.split('\n')
        unique_lines = set()
        duplicate_count = 0
        for line in lines:
            line_clean = line.strip()
            if line_clean and line_clean in unique_lines:
                duplicate_count += 1
            else:
                unique_lines.add(line_clean)
        response_analysis["duplicate_lines"] = duplicate_count
        
        self.logger.info(
            "Full LLM response captured for debugging",
            extra={
                "event_type": "full_llm_response",
                "request_id": request_id,
                "session_id": self.session_id,
                "user_input": user_input[:100] + "..." if len(user_input) > 100 else user_input,
                "provider": provider,
                "response_analysis": response_analysis,
                "full_response": full_response,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @contextmanager
    def trace_request(self, user_input: str, provider: str = None):
        """Context manager for tracing complete requests."""
        context = self.create_request_context(user_input, provider)
        self.log_agent_request_start(context)
        
        try:
            yield context
        except Exception as e:
            self.log_agent_request_end(context, "", success=False, error=str(e))
            raise
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record):
        """Format log record with structured data."""
        # Base log format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        level = record.levelname
        message = record.getMessage()
        
        # Build structured log entry
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "logger": record.name
        }
        
        # Add extra fields if present
        if hasattr(record, 'event_type'):
            log_entry["event_type"] = record.event_type
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
        if hasattr(record, 'request_id'):
            log_entry["request_id"] = record.request_id
        if hasattr(record, 'provider'):
            log_entry["provider"] = record.provider
        if hasattr(record, 'duration_seconds'):
            log_entry["duration_seconds"] = record.duration_seconds
        if hasattr(record, 'success'):
            log_entry["success"] = record.success
        if hasattr(record, 'error'):
            log_entry["error"] = record.error
        if hasattr(record, 'user_input'):
            log_entry["user_input"] = record.user_input
        if hasattr(record, 'full_response'):
            log_entry["full_response"] = record.full_response
        if hasattr(record, 'response_analysis'):
            log_entry["response_analysis"] = record.response_analysis
        if hasattr(record, 'contains_plot_data'):
            log_entry["contains_plot_data"] = record.contains_plot_data
        if hasattr(record, 'plot_data_count'):
            log_entry["plot_data_count"] = record.plot_data_count
        if hasattr(record, 'plot_data_size'):
            log_entry["plot_data_size"] = record.plot_data_size
        if hasattr(record, 'response_length'):
            log_entry["response_length"] = record.response_length
        
        return json.dumps(log_entry, default=str)


# Global logger instance
agent_logger = AgentLogger()