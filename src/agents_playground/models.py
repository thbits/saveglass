"""
Pydantic models for AI Agents Playground
"""
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum
from datetime import datetime


class ProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    BEDROCK = "bedrock"


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    model_config = ConfigDict(extra='allow')
    
    model_name: str = Field(..., description="Name of the model to use")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Temperature for response generation")
    max_tokens: int = Field(default=1000, gt=0, description="Maximum number of tokens in response")
    api_key_env: Optional[str] = Field(default=None, description="Environment variable name for API key")
    region: Optional[str] = Field(default=None, description="AWS region for Bedrock")
    aws_access_key_env: Optional[str] = Field(default=None, description="Environment variable for AWS access key")
    aws_secret_key_env: Optional[str] = Field(default=None, description="Environment variable for AWS secret key")
    aws_region_env: Optional[str] = Field(default=None, description="Environment variable for AWS region")


class AgentSettings(BaseModel):
    """Agent behavior settings."""
    conversation_history_limit: int = Field(default=10, gt=0, description="Maximum conversation history length")
    enable_streaming: bool = Field(default=True, description="Enable streaming responses")
    enable_function_calling: bool = Field(default=True, description="Enable function calling capabilities")


class AgentConfig(BaseModel):
    """Complete agent configuration."""
    default_provider: ProviderType = Field(default=ProviderType.OPENAI, description="Default LLM provider")
    providers: Dict[str, ProviderConfig] = Field(default_factory=dict, description="Provider configurations")
    available_models: Dict[str, List[str]] = Field(default_factory=dict, description="Available models per provider")
    agent_settings: AgentSettings = Field(default_factory=AgentSettings, description="Agent behavior settings")


class PageConfig(BaseModel):
    """Streamlit page configuration."""
    page_title: str = Field(default="AI Agents Playground", description="Page title")
    page_icon: str = Field(default="🤖", description="Page icon")
    layout: str = Field(default="wide", description="Page layout")
    initial_sidebar_state: str = Field(default="expanded", description="Initial sidebar state")


class ThemeConfig(BaseModel):
    """UI theme configuration."""
    primary_color: str = Field(default="#2196f3", description="Primary theme color")
    secondary_color: str = Field(default="#9c27b0", description="Secondary theme color")
    background_color: str = Field(default="#ffffff", description="Background color")
    text_color: str = Field(default="#000000", description="Text color")


class ChatStyling(BaseModel):
    """Chat interface styling configuration."""
    model_config = ConfigDict(extra='allow')
    
    user_message: Dict[str, str] = Field(default_factory=dict, description="User message styling")
    assistant_message: Dict[str, str] = Field(default_factory=dict, description="Assistant message styling")
    message_padding: str = Field(default="1rem", description="Message padding")
    message_border_radius: str = Field(default="0.5rem", description="Message border radius")
    message_margin_bottom: str = Field(default="1rem", description="Message margin bottom")
    input_border_radius: str = Field(default="20px", description="Input border radius")


class SidebarConfig(BaseModel):
    """Sidebar configuration."""
    model_config = ConfigDict(extra='allow')
    
    title: str = Field(default="🔧 Configuration", description="Sidebar title")
    sections: Dict[str, Any] = Field(default_factory=dict, description="Sidebar sections configuration")


class MessagesConfig(BaseModel):
    """UI messages configuration."""
    welcome_message: str = Field(default="Welcome to the AI Agents Playground!", description="Welcome message")
    thinking_spinner: str = Field(default="Thinking...", description="Thinking spinner text")
    error_prefix: str = Field(default="Error: ", description="Error message prefix")
    success_config_update: str = Field(default="Agent configuration updated!", description="Success message for config update")
    warning_no_credentials: str = Field(default="API credentials not configured.", description="Warning for missing credentials")


class ExampleQuestion(BaseModel):
    """Example question configuration."""
    text: str = Field(..., description="The question text")
    description: str = Field(..., description="Description of what the question does")


class ExampleQuestionsConfig(BaseModel):
    """Example questions configuration."""
    enabled: bool = Field(default=True, description="Whether to show example questions")
    title: str = Field(default="💡 Example Questions", description="Section title")
    questions: List[ExampleQuestion] = Field(default_factory=list, description="List of example questions")


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    chart_keywords: List[str] = Field(
        default_factory=lambda: ["chart", "plot", "graph", "visualiz"],
        description="Keywords that trigger visualization generation"
    )
    use_container_width: bool = Field(default=True, description="Use container width for plots")
    default_theme: str = Field(default="plotly", description="Default plotting theme")


class UIConfig(BaseModel):
    """Complete UI configuration."""
    page_config: PageConfig = Field(default_factory=PageConfig, description="Page configuration")
    theme: ThemeConfig = Field(default_factory=ThemeConfig, description="Theme configuration")
    chat_styling: ChatStyling = Field(default_factory=ChatStyling, description="Chat styling configuration")
    sidebar: SidebarConfig = Field(default_factory=SidebarConfig, description="Sidebar configuration")
    messages: MessagesConfig = Field(default_factory=MessagesConfig, description="Messages configuration")
    input_placeholder: str = Field(
        default="Ask me anything... (try: 'generate a sales chart' or 'analyze data trends')",
        description="Input placeholder text"
    )
    example_questions: ExampleQuestionsConfig = Field(default_factory=ExampleQuestionsConfig, description="Example questions configuration")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization configuration")


class SystemPromptsConfig(BaseModel):
    """System prompts configuration."""
    system_prompts: Dict[str, str] = Field(default_factory=dict, description="System prompts by type")


class MockResponsesConfig(BaseModel):
    """Mock responses configuration."""
    mock_responses: Dict[str, str] = Field(default_factory=dict, description="Mock responses by type")


class PromptsConfig(BaseModel):
    """Complete prompts configuration."""
    system_prompts: Dict[str, str] = Field(default_factory=dict, description="System prompts")
    mock_responses: Dict[str, str] = Field(default_factory=dict, description="Mock responses")


class AgentState(BaseModel):
    """State object for LangGraph workflow."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    messages: List[Any] = Field(default_factory=list, description="Conversation messages")
    current_response: str = Field(default="", description="Current response being generated")
    processing_step: str = Field(default="initial", description="Current processing step")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, description="User message")
    prompt_type: str = Field(default="default", description="Type of system prompt to use")
    session_id: Optional[str] = Field(default=None, description="Session identifier")


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str = Field(..., description="Agent response")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional response metadata")


class ConfigUpdateRequest(BaseModel):
    """Configuration update request model."""
    provider_name: Optional[str] = Field(default=None, description="New provider name")
    model_name: Optional[str] = Field(default=None, description="New model name")
    temperature: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="New temperature value")
    max_tokens: Optional[int] = Field(default=None, gt=0, description="New max tokens value")


class AgentCurrentConfig(BaseModel):
    """Current agent configuration response model."""
    provider: str = Field(..., description="Current provider")
    model: str = Field(..., description="Current model")
    temperature: float = Field(..., description="Current temperature")
    max_tokens: int = Field(..., description="Current max tokens")
    conversation_limit: int = Field(..., description="Conversation history limit")


class LogContext(BaseModel):
    """Logging context model."""
    request_id: str = Field(..., description="Request identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Request timestamp")
    user_input: str = Field(..., description="User input (truncated)")
    provider: Optional[str] = Field(default=None, description="LLM provider")
    trace_start: datetime = Field(..., description="Trace start time")


class VisualizationRequest(BaseModel):
    """Visualization request model."""
    prompt: str = Field(..., description="User prompt for visualization")
    chart_type: Optional[str] = Field(default=None, description="Requested chart type")
    data_type: Optional[str] = Field(default=None, description="Type of data to generate")