"""
AI Agent implementation with configurable LLM providers and LangGraph streaming
"""
import os
from typing import Dict, Any, Optional, Union, AsyncIterator, Iterator, List, Callable
from langchain.schema import HumanMessage, SystemMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from agents_playground.config_loader import config_loader
from agents_playground.factory import LLMProviderFactory
from agents_playground.logger import agent_logger
from agents_playground.models import AgentState, AgentCurrentConfig, ConfigUpdateRequest


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
        self.tools = {}  # Registry for custom tools

        # Initialize the LLM client and graph
        self._initialize_client()
        self._initialize_graph()
        self._log_initialization()

    def add_tool(self, name: str, func: Callable, description: str = ""):
        """
        Add a custom tool to the agent.

        Args:
            name: Tool name
            func: Tool function that takes (state, **kwargs) and returns modified state
            description: Tool description for logging
        """
        self.tools[name] = {"func": func, "description": description}
        # Rebuild graph with new tool
        self._initialize_graph()

    def add_tools(self, tools: Dict[str, Dict[str, Any]]):
        """
        Add multiple tools to the agent efficiently (rebuilds graph only once).

        Args:
            tools: Dictionary where keys are tool names and values are dicts containing:
                  - 'func': Tool function that takes (state, **kwargs) and returns modified state
                  - 'description': Tool description for logging (optional)
        """
        for name, tool_info in tools.items():
            if isinstance(tool_info, dict):
                func = tool_info.get('func')
                description = tool_info.get('description', "")
                if func:
                    self.tools[name] = {"func": func, "description": description}
            else:
                # Handle case where tool_info is just the function
                self.tools[name] = {"func": tool_info, "description": ""}
        
        # Rebuild graph only once after adding all tools
        self._initialize_graph()

    def remove_tool(self, name: str):
        """Remove a tool from the agent."""
        if name in self.tools:
            del self.tools[name]
            self._initialize_graph()

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
        """Initialize the LangGraph workflow with custom tools."""
        try:
            workflow = StateGraph(AgentState)

            # Add core nodes
            workflow.add_node("process_input", self._process_input_node)

            # Add custom tool nodes
            for tool_name, tool_info in self.tools.items():
                workflow.add_node(tool_name, tool_info["func"])

            workflow.add_node("generate_response", self._generate_response_node)
            workflow.add_node("finalize", self._finalize_node)

            # Set entry point
            workflow.set_entry_point("process_input")

            # Build edges - tools run after input processing but before response generation
            if self.tools:
                # Connect input to first tool
                first_tool = list(self.tools.keys())[0]
                workflow.add_edge("process_input", first_tool)

                # Chain tools together
                tool_names = list(self.tools.keys())
                for i in range(len(tool_names) - 1):
                    workflow.add_edge(tool_names[i], tool_names[i + 1])

                # Connect last tool to response generation
                workflow.add_edge(tool_names[-1], "generate_response")
            else:
                # Direct connection if no tools
                workflow.add_edge("process_input", "generate_response")

            workflow.add_edge("generate_response", "finalize")
            workflow.add_edge("finalize", END)

            self.graph = workflow.compile(checkpointer=self.memory)

        except Exception as e:
            print(f"Error initializing LangGraph workflow: {e}")
            self.graph = None

    def _process_input_node(self, state: AgentState) -> AgentState:
        """Process user input and prepare for tool execution."""
        state.processing_step = "processing_input"
        state.metadata["timestamp"] = "processing_input"
        state.metadata["tools_available"] = list(self.tools.keys())
        return state

    def _generate_response_node(self, state: AgentState) -> AgentState:
        """Generate response using the LLM."""
        state.processing_step = "generating_response"

        if not self.client:
            if state.messages:
                last_message = state.messages[-1]
                if isinstance(last_message, HumanMessage):
                    state.current_response = self._mock_response(last_message.content)
                else:
                    state.current_response = self._mock_response("Hello")
            else:
                state.current_response = self._mock_response("Hello")
        else:
            try:
                system_prompt = config_loader.get_system_prompt("default")

                # Include tool results in system prompt if available
                if "tool_results" in state.metadata:
                    tool_context = "\nTool Results:\n" + "\n".join(state.metadata["tool_results"])
                    system_prompt += tool_context

                system_message = SystemMessage(content=system_prompt)
                messages = [system_message] + state.messages[-self.conversation_limit:]

                response = self.client(messages)
                state.current_response = response.content

            except Exception as e:
                state.current_response = f"Error generating response: {str(e)}"

        return state

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Finalize the response and update state."""
        state.processing_step = "finalized"
        state.metadata["completed"] = True
        return state

    def _mock_response(self, message: str) -> str:
        """Provide mock responses when LLM client is not available."""
        message_lower = message.lower()

        # Check for tool-specific responses
        for tool_name in self.tools.keys():
            if tool_name.lower() in message_lower:
                return f"Mock response: Executed {tool_name} tool with message: {message}"

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

                agent_logger.log_agent_request_end(context, response, success=True)
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

            state: AgentState = {
                "messages": self.conversation_history.copy(),
                "current_response": "",
                "processing_step": "initial",
                "metadata": {"tool_results": []}
            }

            result = await self.graph.ainvoke(state, config={"configurable": {"thread_id": "default"}})
            response_content = result["current_response"]

            if response_content:
                response_message = AIMessage(content=response_content)
                self.conversation_history.append(response_message)

            return response_content

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
            response = self._chat_legacy(message, prompt_type)
            yield response
            return

        try:
            human_message = HumanMessage(content=message)
            self.conversation_history.append(human_message)

            if len(self.conversation_history) > self.conversation_limit:
                self.conversation_history = self.conversation_history[-self.conversation_limit:]

            state: AgentState = {
                "messages": self.conversation_history.copy(),
                "current_response": "",
                "processing_step": "initial",
                "metadata": {"tool_results": []}
            }

            async for chunk in self.graph.astream(state, config={"configurable": {"thread_id": "default"}}):
                if chunk and "generate_response" in chunk:
                    node_output = chunk["generate_response"]
                    if isinstance(node_output, dict) and node_output.get("current_response"):
                        response = node_output["current_response"]
                        chunk_size = 10
                        for i in range(0, len(response), chunk_size):
                            yield response[i:i + chunk_size]
                            await asyncio.sleep(0.1)

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