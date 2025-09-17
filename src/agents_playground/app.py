"""
AI Agents Playground - Main Streamlit Application
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
import os
import traceback
import asyncio
import re
import json
from datetime import datetime
import uuid
from dotenv import load_dotenv

from agents_playground.agent import ChatAgent
from agents_playground.utils import generate_sample_data, create_plot
from agents_playground.config_loader import config_loader
from agents_playground.logger import agent_logger
from agents_playground.session_storage import session_storage
from agents_playground.langchain_tools import available_tools
from langchain.schema import HumanMessage, AIMessage

# Load environment variables
load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="SaveGlass",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and apply custom CSS for better UI
st.markdown(config_loader.get_css_styles(), unsafe_allow_html=True)

def create_new_session():
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now()
    return {
        "id": session_id,
        "name": f"Chat {timestamp.strftime('%m/%d %H:%M')}",
        "messages": [],
        "created_at": timestamp,
        "last_updated": timestamp
    }

def save_current_session():
    """Save the current session to the sessions list and persist to disk."""
    if "current_session_id" in st.session_state and "chat_sessions" in st.session_state:
        current_id = st.session_state.current_session_id
        for session in st.session_state.chat_sessions:
            if session["id"] == current_id:
                session["messages"] = st.session_state.messages.copy()
                session["last_updated"] = datetime.now()
                # Persist the updated session to disk
                session_storage.save_single_session(session)
                break

def load_session(session_id: str):
    """Load a specific session."""
    save_current_session()  # Save current session before switching
    
    for session in st.session_state.chat_sessions:
        if session["id"] == session_id:
            st.session_state.messages = session["messages"].copy()
            st.session_state.current_session_id = session_id
            st.session_state.agent.clear_history()
            # Rebuild agent conversation history
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    st.session_state.agent.conversation_history.append(
                        HumanMessage(content=msg["content"])
                    )
                elif msg["role"] == "assistant":
                    st.session_state.agent.conversation_history.append(
                        AIMessage(content=msg["content"])
                    )
            break

def get_session_cluster():
    """Get the current cluster from session state."""
    return getattr(st.session_state, 'current_cluster', None)

def set_session_cluster(cluster_name):
    """Set the current cluster in session state."""
    st.session_state.current_cluster = cluster_name
    st.session_state.cluster_asked = True

def has_asked_for_cluster():
    """Check if we've already asked for cluster in this session."""
    return getattr(st.session_state, 'cluster_asked', False)

def initialize_session_state():
    """Initialize session state variables."""
    try:
        # Initialize chat sessions storage - load from persistent storage
        if "chat_sessions" not in st.session_state:
            # Load existing sessions from persistent storage
            persisted_sessions = session_storage.load_sessions()
            st.session_state.chat_sessions = persisted_sessions
        
        # Initialize current session
        if "current_session_id" not in st.session_state:
            if st.session_state.chat_sessions:
                # Use the most recent session as current
                latest_session = max(st.session_state.chat_sessions, key=lambda s: s.get("last_updated", s.get("created_at", datetime.min)))
                st.session_state.current_session_id = latest_session["id"]
                st.session_state.messages = latest_session.get("messages", [])
            else:
                # Create new session if no persisted sessions exist
                new_session = create_new_session()
                st.session_state.chat_sessions.append(new_session)
                st.session_state.current_session_id = new_session["id"]
                st.session_state.messages = []
                # Persist the new session immediately
                session_storage.save_single_session(new_session)
        
        # Initialize cluster memory for session
        if "current_cluster" not in st.session_state:
            st.session_state.current_cluster = None
        if "cluster_asked" not in st.session_state:
            st.session_state.cluster_asked = False
            
        # Initialize agent
        if "agent" not in st.session_state:
            agent_logger.info("Initializing new agent instance")
            st.session_state.agent = ChatAgent()
            
            # Add AWS cost tools to the agent using the LangChain tools
            # The tools are now automatically loaded with create_react_agent via available_tools
            # No need to manually add tools as they're included in the agent initialization
            
            agent_logger.info("Added AWS cost tools to agent")
            
    except Exception as e:
        agent_logger.log_error(e, {"context": "session_initialization"})
        st.error(f"Failed to initialize session: {str(e)}")
        st.stop()

def extract_plots_from_response(response: str) -> tuple[str, list]:
    """Extract plot data from agent response and return clean text with plots."""
    plots = []
    clean_response = response
    
    # Log the extraction attempt
    agent_logger.info(
        "Starting plot extraction from response",
        response_length=len(response),
        contains_plot_markers="[PLOT_DATA]" in response and "[/PLOT_DATA]" in response
    )
    
    # Find all plot data markers
    plot_pattern = r'\[PLOT_DATA\](.*?)\[/PLOT_DATA\]'
    plot_matches = re.findall(plot_pattern, response, re.DOTALL)
    
    agent_logger.info(f"Found {len(plot_matches)} plot data blocks")
    
    for i, plot_json in enumerate(plot_matches):
        try:
            plot_json_stripped = plot_json.strip()
            agent_logger.info(
                f"Processing plot {i+1}",
                plot_data_length=len(plot_json_stripped),
                plot_preview=plot_json_stripped[:100] + "..." if len(plot_json_stripped) > 100 else plot_json_stripped
            )
            
            # Parse JSON and create plotly figure
            plot_dict = json.loads(plot_json_stripped)
            fig = go.Figure(plot_dict)
            plots.append(fig)
            
            agent_logger.info(
                f"Successfully created plot {i+1}",
                plot_type=plot_dict.get('data', [{}])[0].get('type', 'unknown') if plot_dict.get('data') else 'unknown',
                has_layout=bool(plot_dict.get('layout')),
                data_points=len(plot_dict.get('data', []))
            )
            
        except json.JSONDecodeError as e:
            agent_logger.log_error(
                e, {
                    "context": "plot_json_parsing", 
                    "plot_index": i,
                    "plot_json_length": len(plot_json),
                    "plot_json_preview": plot_json[:200],
                    "json_error": str(e)
                }
            )
        except Exception as e:
            agent_logger.log_error(
                e, {
                    "context": "plot_figure_creation", 
                    "plot_index": i,
                    "plot_json": plot_json[:100]
                }
            )
    
    # Remove plot data from response text
    clean_response = re.sub(plot_pattern, '', response, flags=re.DOTALL).strip()
    
    agent_logger.info(
        "Plot extraction completed",
        plots_extracted=len(plots),
        clean_response_length=len(clean_response),
        original_response_length=len(response)
    )
    
    return clean_response, plots


def display_chat_message(role: str, content: str, plots: list = None):
    """Display a chat message with optional plots."""
    css_class = "user-message" if role == "user" else "assistant-message"
    
    with st.container():
        st.markdown(f'<div class="chat-message {css_class}">', unsafe_allow_html=True)
        st.markdown(f"**{role.title()}:** {content}")
        
        # Display plots if any
        if plots:
            for plot in plots:
                st.plotly_chart(plot, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main application function."""
    initialize_session_state()
    
    # Sidebar for configuration
    with st.sidebar:
        
        # Chat Sessions History
        st.subheader("💬 Chat Sessions")
        
        # New session button
        if st.button("➕ New Chat Session"):
            save_current_session()  # Save current session
            new_session = create_new_session()
            st.session_state.chat_sessions.append(new_session)
            st.session_state.current_session_id = new_session["id"]
            st.session_state.messages = []
            st.session_state.agent.clear_history()
            # Persist the new session immediately
            session_storage.save_single_session(new_session)
            st.rerun()
        
        # Display existing sessions
        if st.session_state.chat_sessions:
            st.markdown("**Recent Sessions:**")
            for session in reversed(st.session_state.chat_sessions[-10:]):  # Show last 10 sessions
                is_current = session["id"] == st.session_state.current_session_id
                
                # Session display with timestamp and message count
                msg_count = len(session["messages"])
                time_str = session["last_updated"].strftime("%m/%d %H:%M")
                session_label = f"{session['name']} ({msg_count} msgs) - {time_str}"
                
                if is_current:
                    st.markdown(f"🟢 **{session_label}** *(current)*")
                else:
                    if st.button(session_label, key=f"session_{session['id']}"):
                        load_session(session["id"])
                        st.rerun()
        
        st.divider()
        
        # Agent settings
        st.subheader("Agent Settings")
        
        # Provider selection
        available_providers = st.session_state.agent.get_available_providers()
        current_provider = st.session_state.agent.provider_name
        
        provider_name = st.selectbox(
            "LLM Provider",
            available_providers,
            index=available_providers.index(current_provider) if current_provider in available_providers else 0
        )
        
        # Model selection based on provider
        available_models = config_loader.get_available_models(provider_name)
        if available_models:
            current_config = st.session_state.agent.get_current_config()
            current_model = current_config.model
            
            # If the current model is not available for the selected provider, default to first model
            default_index = 0
            if current_model in available_models:
                default_index = available_models.index(current_model)
            
            model_name = st.selectbox(
                "Model",
                available_models,
                index=default_index
            )
        else:
            model_name = st.text_input("Model Name", value="gpt-3.5-turbo")
        
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_tokens = st.number_input("Max Tokens", 100, 4000, 1000)
        
        # Update agent configuration
        if st.button("Update Agent Config"):
            try:
                agent_logger.info("User updating agent configuration", 
                                provider=provider_name, 
                                model=model_name, 
                                temperature=temperature, 
                                max_tokens=max_tokens)
                
                st.session_state.agent.update_config(
                    provider_name=provider_name,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.success("Agent configuration updated!")
                st.rerun()
            except Exception as e:
                agent_logger.log_error(e, {"context": "config_update_ui"})
                st.error(f"Failed to update configuration: {str(e)}")
                st.rerun()
        
        # Display current configuration
        st.subheader("Current Config")
        config = st.session_state.agent.get_current_config()
        st.json(config.model_dump())
        
        # Clear chat
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.session_state.agent.clear_history()
            save_current_session()  # Save cleared session to persistent storage
            st.rerun()
        
        # Provider Configuration Status
        st.subheader("Provider Status")
        if provider_name == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                st.success("✅ OpenAI API key configured")
            else:
                st.warning("⚠️ OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        elif provider_name == "bedrock":
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            st.info(f"AWS Region: {aws_region}")
            
            if os.getenv("AWS_ACCESS_KEY_ID"):
                st.success("✅ AWS credentials configured")
            else:
                st.warning("⚠️ AWS credentials not configured. Using default profile if available.")
        
        st.divider()
        
        # Theme toggle at the bottom
        st.subheader("🎨 Theme")
        
        # Initialize theme in session state
        if "theme_mode" not in st.session_state:
            st.session_state.theme_mode = "light"
        
        # Theme toggle button with emoji
        current_theme = st.session_state.theme_mode
        button_emoji = "🌙" if current_theme == "light" else "☀️"
        button_text = f"{button_emoji} Switch to {'Dark' if current_theme == 'light' else 'Light'} Mode"
        
        if st.button(button_text, help="Toggle between light and dark themes", key="theme_toggle_button", use_container_width=True):
            if current_theme == "light":
                # Switch to dark theme using Glassbox dark colors
                st._config.set_option("theme.base", "dark")
                st._config.set_option("theme.backgroundColor", "#1E1E1E")
                st._config.set_option("theme.primaryColor", "#7059FF")
                st._config.set_option("theme.secondaryBackgroundColor", "#2D2B4A")
                st._config.set_option("theme.textColor", "#E0E0E0")
                st.session_state.theme_mode = "dark"
            else:
                # Switch to light theme using Glassbox light colors
                st._config.set_option("theme.base", "light")
                st._config.set_option("theme.backgroundColor", "#FFFFFF")
                st._config.set_option("theme.primaryColor", "#251B9C")
                st._config.set_option("theme.secondaryBackgroundColor", "#F8F7FF")
                st._config.set_option("theme.textColor", "#2D2D2D")
                st.session_state.theme_mode = "light"
            
            st.rerun()
        
        # Display current theme
        st.caption(f"Current theme: {current_theme.title()}")
    
    # Main chat interface with logo
    # Create a container for the logo and title with better spacing
    logo_path = os.path.join(os.path.dirname(__file__), "../../resources/ui/glassbox-logo.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "r") as f:
            logo_svg = f.read()
        # Display logo and title in a single line with better alignment
        st.markdown(f'''<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;"><div style="width: 50px; height: 50px; flex-shrink: 0;">{logo_svg}</div><h1 style="margin: 0; font-size: 3rem; font-weight: 600; color: inherit;">SaveGlass</h1></div>''', unsafe_allow_html=True)
    else:
        st.title("🔍 SaveGlass")
    
    st.markdown("Welcome to SaveGlass! Ask questions about your AWS costs, request data analysis, or generate plots.")
    
    # Display example questions
    try:
        ui_config = config_loader.get_ui_config()
        if ui_config.example_questions.enabled and ui_config.example_questions.questions:
            st.markdown(f"### {ui_config.example_questions.title}")
            
            # Create columns for example questions
            cols = st.columns(len(ui_config.example_questions.questions))
            
            for idx, question in enumerate(ui_config.example_questions.questions):
                with cols[idx]:
                    # Create button with question text
                    if st.button(
                        question.text,
                        key=f"example_q_{idx}",
                        help=question.description,
                        use_container_width=True
                    ):
                        # Set the question text in session state to be used by chat input
                        st.session_state.example_question_selected = question.text
                        st.rerun()
            
            st.divider()
    except Exception as e:
        agent_logger.log_error(e, {"context": "example_questions_display"})
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"], 
            message["content"], 
            message.get("plots", [])
        )
    
    # Chat input
    # Check if an example question was selected
    if "example_question_selected" in st.session_state:
        prompt = st.session_state.example_question_selected
        del st.session_state.example_question_selected  # Clean up
        
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_current_session()  # Save session after adding user message
        display_chat_message("user", prompt)
    elif prompt := st.chat_input("Ask me anything... (try: 'generate a sales chart' or 'analyze data trends')"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        save_current_session()  # Save session after adding user message
        display_chat_message("user", prompt)
    else:
        prompt = None
    
    if prompt:
        # Generate response
        with st.spinner("Thinking..."):
            try:
                agent_logger.info("Processing user chat request", user_input=prompt[:100])
                
                # Get agent response with streaming
                final_response = ""
                async def stream_response():
                    nonlocal final_response
                    async for chunk in st.session_state.agent.chat_stream(prompt):
                        # Capture the final complete response when plot generation is complete
                        if chunk not in ["PLOT_GENERATION_STARTED", "PLOT_GENERATION_COMPLETE"]:
                            # Only set final_response for the complete response, not incremental chunks
                            if "[PLOT_DATA]" in chunk and "[/PLOT_DATA]" in chunk:
                                final_response = chunk
                        yield chunk
                
                # Display streaming response using consistent styling
                response_placeholder = st.empty()
                streamed_content = ""
                
                async def display_stream():
                    nonlocal streamed_content
                    generating_plot = False
                    async for chunk in stream_response():
                        # Handle special plot generation signals
                        if chunk == "PLOT_GENERATION_STARTED":
                            generating_plot = True
                            # Update display to show plot generation status
                            clean_streamed_content, _ = extract_plots_from_response(streamed_content)
                            with response_placeholder.container():
                                display_chat_message("assistant", clean_streamed_content + "\n\n📊 *Generating plot...*")
                            continue
                        elif chunk == "PLOT_GENERATION_COMPLETE":
                            generating_plot = False
                            # The next chunk will be the full response with plot data
                            continue
                        elif generating_plot:
                            # During plot generation, we receive the full response - replace streamed_content
                            streamed_content = chunk
                            break
                        else:
                            # Normal streaming - accumulate content
                            streamed_content += chunk
                            # Extract and display clean content without plot data during streaming
                            clean_streamed_content, _ = extract_plots_from_response(streamed_content)
                            with response_placeholder.container():
                                display_chat_message("assistant", clean_streamed_content)
                        
                asyncio.run(display_stream())
                    
                # Use the final response if available, otherwise use streamed content
                response = final_response if final_response else streamed_content
                
                # Extract plots from tool responses
                clean_response, tool_plots = extract_plots_from_response(response)
                
                # Check if response includes plot generation request (for user visualization requests)
                plots = tool_plots.copy()  # Start with plots from tools
                is_visualization_request = any(keyword in prompt.lower() for keyword in ["chart", "plot", "graph", "visualiz"])
                
                if is_visualization_request and not tool_plots:  # Only generate sample plots if no tool plots
                    try:
                        agent_logger.info("Processing visualization request", prompt=prompt[:100])
                        
                        # Generate sample plot based on the request
                        sample_data = generate_sample_data(prompt)
                        plot = create_plot(sample_data, prompt)
                        
                        if plot:
                            plots.append(plot)
                            chart_type = "auto-detected"
                            agent_logger.log_visualization_request(prompt, chart_type, True)
                        else:
                            agent_logger.log_visualization_request(prompt, "failed", False)
                            
                    except Exception as viz_error:
                        agent_logger.log_error(viz_error, {
                            "context": "visualization_generation",
                            "prompt": prompt[:100]
                        })
                        agent_logger.log_visualization_request(prompt, "error", False)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": clean_response,  # Use clean response without plot data
                    "plots": plots
                })
                
                # Save current session after adding response
                save_current_session()
                
                # Display plots if any (response was already streamed above)
                if plots:
                    for plot in plots:
                        st.plotly_chart(plot, use_container_width=True)
                
            except Exception as e:
                agent_logger.log_error(e, {
                    "context": "chat_processing",
                    "user_input": prompt[:100],
                    "traceback": traceback.format_exc()
                })
                
                error_message = f"I apologize, but I encountered an error processing your request: {str(e)}"
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })
                save_current_session()  # Save session after error message
                display_chat_message("assistant", error_message)
        
        st.rerun()

if __name__ == "__main__":
    try:
        agent_logger.info("Starting SaveGlass application")
        main()
    except Exception as e:
        agent_logger.log_error(e, {
            "context": "application_startup",
            "traceback": traceback.format_exc()
        })
        st.error("Fatal error occurred during application startup. Please check the logs.")
        st.stop()