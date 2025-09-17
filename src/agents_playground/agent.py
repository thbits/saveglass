"""
AI Agent implementation with configurable LLM providers and LangGraph streaming
"""
import os
from typing import Dict, Any, Optional, Union, AsyncIterator, Iterator, List, Callable
from langchain.schema import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from agents_playground.config_loader import config_loader
from agents_playground.factory import LLMProviderFactory
from agents_playground.logger import agent_logger
from agents_playground.models import AgentCurrentConfig, ConfigUpdateRequest
from agents_playground.langchain_tools import available_tools


class BaseAgent:
    """
    Base configurable agent that supports multiple LLM providers with LangGraph.
    Can be extended with custom tools and behaviors.
    """

    def __init__(self, provider_name: Optional[str] = None, model_name: Optional[str] = None):
        """
        Initialize the base agent with configurable provider.

        Args:
            provider_name: LLM provider to use (defaults to config default)
            model_name: Specific model to use (overrides config default)
        """
        self.config = config_loader.get_agent_config()
        self.provider_name = provider_name or config_loader.get_default_provider()
        self.provider_config = config_loader.get_provider_config(self.provider_name)

        # Override model if specified and provider config exists
        if model_name and self.provider_config:
            config_dict = self.provider_config.model_dump()
            config_dict["model_name"] = model_name
            from agents_playground.models import ProviderConfig
            self.provider_config = ProviderConfig(**config_dict)

        self.client = None
        self.conversation_history = []
        self.conversation_limit = self.config.agent_settings.conversation_history_limit

        # Initialize LangGraph components
        self.memory = MemorySaver()
        self.graph = None
        # Initialize the LLM client and graph
        self._initialize_client()
        self._initialize_graph()
        self._log_initialization()

    def add_tool(self, name: str, func: Callable, description: str = ""):
        """
        Add a custom tool to the agent (deprecated - tools are now configured via available_tools).
        
        Args:
            name: Tool name  
            func: Tool function
            description: Tool description
        """
        agent_logger.info(f"add_tool called but tools are now managed via available_tools in langchain_tools.py")

    def add_tools(self, tools: Dict[str, Dict[str, Any]]):
        """
        Add multiple tools to the agent (deprecated - tools are now configured via available_tools).
        
        Args:
            tools: Dictionary of tools
        """
        agent_logger.info(f"add_tools called with {len(tools)} tools but tools are now managed via available_tools in langchain_tools.py")

    def remove_tool(self, name: str):
        """Remove a tool from the agent (deprecated - tools are now configured via available_tools)."""
        agent_logger.info(f"remove_tool called for {name} but tools are now managed via available_tools in langchain_tools.py")

    def _initialize_client(self):
        """Initialize the LLM client based on the configured provider."""
        try:
            self.client = LLMProviderFactory.create_provider(self.provider_name, self.provider_config)
            if not self.client:
                print(f"Warning: Could not initialize {self.provider_name} client.")
                print("Using mock responses for development.")
        except Exception as e:
            print(f"Error initializing {self.provider_name} client: {e}")
            print("Using mock responses for development.")
            self.client = None

    def _initialize_graph(self):
        """Initialize the LangGraph agent using create_react_agent."""
        try:
            if not self.client:
                print("Warning: No LLM client available, graph will not be initialized.")
                self.graph = None
                return
                
            # Get available tools (initially empty, will be populated via add_tools)
            tools_list = list(available_tools)
            
            # Create the react agent using the prebuilt function
            # Set recursion_limit to prevent infinite loops
            self.graph = create_react_agent(
                self.client,
                tools_list,
                checkpointer=self.memory
            )
            
            # Configure recursion limit for the graph to prevent infinite loops
            if hasattr(self.graph, 'config'):
                self.graph.config = self.graph.config or {}
                self.graph.config['recursion_limit'] = 10
            
            agent_logger.info(f"LangGraph agent initialized with {len(tools_list)} tools")

        except Exception as e:
            print(f"Error initializing LangGraph workflow: {e}")
            agent_logger.log_error(e, {"context": "graph_initialization"})
            self.graph = None


    def _mock_response(self, message: str) -> str:
        """Provide mock responses when LLM client is not available."""
        message_lower = message.lower()

        # Check for tool-specific responses based on available tools
        tool_keywords = ["cost", "costs", "aws", "billing", "expense", "last month", "current month"]
        if any(keyword in message_lower for keyword in tool_keywords):
            return f"Mock response: Would execute AWS cost tools for query: {message}"

        # Default mock responses
        ui_config = config_loader.get_ui_config()
        chart_keywords = ui_config.visualization.chart_keywords

        if any(keyword in message_lower for keyword in chart_keywords):
            template = config_loader.get_mock_response("visualization_request")
            return template.format(message=message)
        elif "hello" in message_lower or "hi" in message_lower:
            return config_loader.get_mock_response("greeting")
        elif "help" in message_lower:
            return config_loader.get_mock_response("help")
        else:
            template = config_loader.get_mock_response("fallback")
            return template.format(message=message)

    def _log_initialization(self):
        """Log agent initialization."""
        try:
            current_config = self.get_current_config()
            agent_logger.log_agent_initialization(
                provider=self.provider_name,
                model=self.provider_config.model_name if self.provider_config else "unknown",
                config=current_config
            )
        except Exception as e:
            agent_logger.log_error(e, {"context": "agent_initialization"})

    def update_config(self, provider_name: Optional[str] = None, model_name: Optional[str] = None,
                     temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """Update the agent configuration."""
        old_config = self.get_current_config()

        if provider_name:
            self.provider_name = provider_name
            self.provider_config = config_loader.get_provider_config(self.provider_name)

        if self.provider_config and (model_name or temperature is not None or max_tokens is not None):
            config_dict = self.provider_config.model_dump()

            if model_name:
                config_dict["model_name"] = model_name
            if temperature is not None:
                config_dict["temperature"] = temperature
            if max_tokens is not None:
                config_dict["max_tokens"] = max_tokens

            from agents_playground.models import ProviderConfig
            self.provider_config = ProviderConfig(**config_dict)

        self._initialize_client()

        try:
            new_config = self.get_current_config()
            agent_logger.log_config_update(old_config.model_dump(), new_config.model_dump())
        except Exception as e:
            agent_logger.log_error(e, {"context": "config_update"})

    def get_available_models(self) -> list:
        """Get available models for the current provider."""
        return config_loader.get_available_models(self.provider_name)

    def get_available_providers(self) -> list:
        """Get list of available providers."""
        agent_config = config_loader.get_agent_config()
        return list(agent_config.providers.keys())

    async def chat(self, message: str, prompt_type: str = "default") -> str:
        """Send a message to the agent and get a response."""
        with agent_logger.trace_request(message, self.provider_name) as context:
            try:
                if self.graph:
                    response = await self._chat_with_graph(message, prompt_type)
                else:
                    response = await self._chat_legacy(message, prompt_type)

                # Log full response for debugging if it contains plot data or if debugging is enabled
                log_full = "[PLOT_DATA]" in response or os.getenv("DEBUG_FULL_RESPONSES", "false").lower() == "true"
                agent_logger.log_agent_request_end(context, response, success=True, log_full_response=log_full)
                
                # Also log detailed response analysis
                agent_logger.log_full_llm_response(context["request_id"], message, response, self.provider_name)
                
                return response

            except Exception as e:
                agent_logger.log_agent_request_end(context, "", success=False, error=str(e))
                agent_logger.log_error(e, {"context": "chat_request", "message": message})
                raise

    async def _chat_with_graph(self, message: str, prompt_type: str = "default") -> str:
        """Chat using LangGraph workflow."""
        try:
            human_message = HumanMessage(content=message)
            self.conversation_history.append(human_message)

            if len(self.conversation_history) > self.conversation_limit:
                self.conversation_history = self.conversation_history[-self.conversation_limit:]

            # Add system prompt to messages
            system_prompt = config_loader.get_system_prompt(prompt_type)
            messages = [SystemMessage(content=system_prompt)] + self.conversation_history

            # create_react_agent expects {"messages": [...]} format
            state = {"messages": messages}

            result = await self.graph.ainvoke(state, config={"configurable": {"thread_id": "default"}, "recursion_limit": 10})
            
            # Extract response from the result messages
            if "messages" in result and result["messages"]:
                last_message = result["messages"][-1]
                if hasattr(last_message, 'content'):
                    response_content = last_message.content
                    response_message = AIMessage(content=response_content)
                    self.conversation_history.append(response_message)
                    return response_content

            return "No response generated"

        except Exception as e:
            return f"Error in graph processing: {str(e)}"

    async def _chat_legacy(self, message: str, prompt_type: str = "default") -> str:
        """Legacy chat method without graph."""
        if not self.client:
            return self._mock_response(message)

        try:
            system_prompt = config_loader.get_system_prompt(prompt_type)
            system_message = SystemMessage(content=system_prompt)
            human_message = HumanMessage(content=message)

            self.conversation_history.append(human_message)

            if len(self.conversation_history) > self.conversation_limit:
                self.conversation_history = self.conversation_history[-self.conversation_limit:]

            # Use ainvoke for async call if available, fallback to synchronous
            if hasattr(self.client, 'ainvoke'):
                response = await self.client.ainvoke([system_message] + self.conversation_history[-self.conversation_limit:])
            else:
                response = self.client([system_message] + self.conversation_history[-self.conversation_limit:])
            self.conversation_history.append(response)

            return response.content

        except Exception as e:
            return f"Error communicating with AI service: {str(e)}"

    async def chat_stream(self, message: str, prompt_type: str = "default") -> AsyncIterator[str]:
        """Stream a response from the agent using LangGraph."""
        if not self.graph:
            response = await self._chat_legacy(message, prompt_type)
            yield response
            return

        try:
            human_message = HumanMessage(content=message)
            self.conversation_history.append(human_message)

            if len(self.conversation_history) > self.conversation_limit:
                self.conversation_history = self.conversation_history[-self.conversation_limit:]

            # Add system prompt to messages
            system_prompt = config_loader.get_system_prompt(prompt_type)
            messages = [SystemMessage(content=system_prompt)] + self.conversation_history

            # create_react_agent expects {"messages": [...]} format
            state = {"messages": messages}

            full_response = ""
            plot_detected = False
            async for chunk in self.graph.astream(state, config={"configurable": {"thread_id": "default"}, "recursion_limit": 10}):
                # create_react_agent streams differently - look for messages in chunk
                for key, value in chunk.items():
                    if isinstance(value, dict) and "messages" in value:
                        messages = value["messages"]
                        if messages and hasattr(messages[-1], 'content'):
                            # Stream the content in chunks
                            content = messages[-1].content
                            if content and content != full_response:
                                new_content = content[len(full_response):]
                                full_response = content
                                
                                # Check if we encounter a plot marker - if so, stop streaming
                                if "[PLOT_DATA]" in new_content and not plot_detected:
                                    plot_detected = True
                                    # Stream content up to the plot marker
                                    plot_start = new_content.find("[PLOT_DATA]")
                                    content_before_plot = new_content[:plot_start]
                                    if content_before_plot:
                                        chunk_size = 10
                                        for i in range(0, len(content_before_plot), chunk_size):
                                            yield content_before_plot[i:i + chunk_size]
                                            await asyncio.sleep(0.1)
                                    # Signal that we're generating a plot and stop streaming
                                    yield "PLOT_GENERATION_STARTED"
                                    break
                                elif not plot_detected:
                                    # Normal streaming without plot data
                                    chunk_size = 10
                                    for i in range(0, len(new_content), chunk_size):
                                        yield new_content[i:i + chunk_size]
                                        await asyncio.sleep(0.1)

            # If plot was detected, yield the final complete response
            if plot_detected:
                yield "PLOT_GENERATION_COMPLETE"
                yield full_response

            # Add final response to conversation history
            if full_response:
                response_message = AIMessage(content=full_response)
                self.conversation_history.append(response_message)
                
                # Log full streaming response for debugging
                import uuid
                request_id = str(uuid.uuid4())
                agent_logger.log_full_llm_response(request_id, message, full_response, self.provider_name)

        except Exception as e:
            yield f"Error in streaming: {str(e)}"

    def clear_history(self):
        """Clear the conversation history."""
        self.conversation_history = []

    def get_current_config(self) -> AgentCurrentConfig:
        """Get current agent configuration."""
        return AgentCurrentConfig(
            provider=self.provider_name,
            model=self.provider_config.model_name if self.provider_config else "unknown",
            temperature=self.provider_config.temperature if self.provider_config else 0.7,
            max_tokens=self.provider_config.max_tokens if self.provider_config else 1000,
            conversation_limit=self.conversation_limit
        )


class ChatAgent(BaseAgent):
    """
    Default chat agent - maintains backward compatibility.
    """
    pass