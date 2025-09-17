"""
Authentication utilities and middleware for Streamlit application.
"""
import streamlit as st
from functools import wraps
from typing import Optional, Callable, Any
from datetime import datetime

from agents_playground.auth_manager import auth_manager
from agents_playground.models import AuthSession, UserPermissions, UserRole
from agents_playground.logger import agent_logger


def get_current_session() -> Optional[AuthSession]:
    """Get the current authenticated session."""
    if "auth_session" in st.session_state:
        session_id = st.session_state.auth_session.get("session_id")
        if session_id:
            return auth_manager.validate_session(session_id)
    return None


def is_authenticated() -> bool:
    """Check if the current user is authenticated."""
    session = get_current_session()
    return session is not None


def get_current_user_permissions() -> Optional[UserPermissions]:
    """Get the current user's permissions."""
    session = get_current_session()
    return session.permissions if session else None


def get_current_user_role() -> Optional[UserRole]:
    """Get the current user's role."""
    session = get_current_session()
    return session.role if session else None


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_authenticated():
            show_login_page()
            st.stop()
        return func(*args, **kwargs)
    return wrapper


def require_permission(permission_check: Callable[[UserPermissions], bool]):
    """Decorator to require specific permissions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session = get_current_session()
            if not session:
                show_login_page()
                st.stop()
            
            if not permission_check(session.permissions):
                st.error("🚫 You don't have permission to access this feature.")
                st.info(f"Required permissions not met. Your role: {session.role}")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def require_role(min_role: UserRole):
    """Decorator to require a minimum role level."""
    role_hierarchy = {
        UserRole.GUEST: 0,
        UserRole.USER: 1,
        UserRole.POWER_USER: 2,
        UserRole.ADMIN: 3
    }
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            session = get_current_session()
            if not session:
                show_login_page()
                st.stop()
            
            user_level = role_hierarchy.get(session.role, 0)
            required_level = role_hierarchy.get(min_role, 3)
            
            if user_level < required_level:
                st.error(f"🚫 This feature requires {min_role} role or higher.")
                st.info(f"Your current role: {session.role}")
                st.stop()
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def login_user(username: str, password: str) -> bool:
    """Attempt to login a user."""
    from agents_playground.models import UserLogin
    
    try:
        login_request = UserLogin(username=username, password=password)
        auth_response = auth_manager.authenticate_user(login_request)
        
        if auth_response.success:
            # Store session information
            st.session_state.auth_session = {
                "session_id": auth_response.session_id,
                "user": auth_response.user,
                "permissions": auth_response.permissions,
                "login_time": datetime.utcnow().isoformat()
            }
            
            # Clear any existing chat data to start fresh for the new user
            if "messages" in st.session_state:
                del st.session_state.messages
            if "chat_sessions" in st.session_state:
                del st.session_state.chat_sessions
            if "current_session_id" in st.session_state:
                del st.session_state.current_session_id
            
            # Show personalized welcome message for special users
            if username.lower() == "yarong":
                st.success("👑 It's nice having you here, captain G")
            else:
                st.success("Login successful!")
            
            agent_logger.info(f"User {username} logged in successfully")
            return True
        else:
            st.error(f"Login failed: {auth_response.message}")
            return False
            
    except Exception as e:
        agent_logger.log_error(e, {"context": "login_user", "username": username})
        st.error(f"Login error: {str(e)}")
        return False


def logout_user():
    """Logout the current user."""
    try:
        if "auth_session" in st.session_state:
            session_id = st.session_state.auth_session.get("session_id")
            if session_id:
                auth_manager.logout_user(session_id)
        
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        agent_logger.info("User logged out successfully")
        st.success("Logged out successfully!")
        st.rerun()
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "logout_user"})
        st.error(f"Logout error: {str(e)}")


def show_login_page():
    """Display the login page."""
    st.title("🔐 Login to AI Agents Playground")
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        with st.form("login_form"):
            st.markdown("### Please sign in to continue")
            
            username = st.text_input("Username or Email", placeholder="Enter your username or email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            login_button = st.form_submit_button("Login", use_container_width=True)
            
            if login_button:
                if username and password:
                    if login_user(username, password):
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                else:
                    st.error("Please enter both username and password")
        
        st.markdown("---")
        
        # Show default admin credentials for demo
        with st.expander("🔧 Demo Credentials"):
            st.info("""
            **Default Admin Account:**
            - Username: `admin`
            - Password: `admin123`
            
            *Note: Change the default password in production!*
            """)
        
        # Role information
        with st.expander("👤 User Roles & Permissions"):
            st.markdown("""
            **Admin:**
            - Full access to all features
            - Can access AWS tools and system information
            - Can manage other users
            - Can change LLM configurations
            
            **Power User:**
            - Access to AWS analysis tools
            - Can change LLM configurations
            - Cannot manage users or view system info
            
            **User:**
            - Basic chat functionality
            - Limited session length and daily requests
            - Cannot access AWS tools or change configurations
            
            **Guest:**
            - Very limited access
            - Basic chat only with strict limits
            """)


def check_prompt_permissions(prompt: str, prompt_type: str = "basic") -> bool:
    """Check if the current user has permission to use a specific prompt type."""
    session = get_current_session()
    if not session:
        return False
    
    # Check if the prompt type is allowed for this user
    if prompt_type not in session.permissions.allowed_prompt_types:
        return False
    
    # Check for AWS-related prompts
    aws_keywords = ["aws", "cost", "billing", "ec2", "s3", "lambda", "cloudformation"]
    if any(keyword in prompt.lower() for keyword in aws_keywords):
        if not session.permissions.can_access_aws_tools:
            return False
    
    return True


def get_user_session_limits() -> dict:
    """Get the current user's session limits."""
    session = get_current_session()
    if not session:
        return {"max_session_length": 5, "max_daily_requests": 10}
    
    return {
        "max_session_length": session.permissions.max_session_length,
        "max_daily_requests": session.permissions.max_daily_requests,
        "role": session.role,
        "username": session.username
    }


def can_change_llm_config() -> bool:
    """Check if the current user can change LLM configuration."""
    session = get_current_session()
    return session and session.permissions.can_change_llm_config


def can_access_aws_tools() -> bool:
    """Check if the current user can access AWS tools."""
    session = get_current_session()
    return session and session.permissions.can_access_aws_tools


def can_view_system_info() -> bool:
    """Check if the current user can view system information."""
    session = get_current_session()
    return session and session.permissions.can_view_system_info


def can_export_sessions() -> bool:
    """Check if the current user can export chat sessions."""
    session = get_current_session()
    return session and session.permissions.can_export_sessions


def show_user_info_sidebar():
    """Show user information in the sidebar."""
    session = get_current_session()
    if not session:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("👤 User Info")
    
    # User details
    st.sidebar.markdown(f"**User:** {session.username}")
    st.sidebar.markdown(f"**Role:** {session.role}")
    
    # Session info
    session_age = datetime.utcnow() - session.created_at
    hours_left = (session.expires_at - datetime.utcnow()).total_seconds() / 3600
    
    st.sidebar.markdown(f"**Session:** {session_age.seconds // 3600}h {(session_age.seconds % 3600) // 60}m active")
    st.sidebar.markdown(f"**Expires:** {hours_left:.1f}h remaining")
    
    # Permissions summary
    with st.sidebar.expander("🔑 Your Permissions"):
        perms = session.permissions
        st.write(f"AWS Tools: {'✅' if perms.can_access_aws_tools else '❌'}")
        st.write(f"LLM Config: {'✅' if perms.can_change_llm_config else '❌'}")
        st.write(f"System Info: {'✅' if perms.can_view_system_info else '❌'}")
        st.write(f"Export Sessions: {'✅' if perms.can_export_sessions else '❌'}")
        st.write(f"Max Session Length: {perms.max_session_length}")
        st.write(f"Daily Requests: {perms.max_daily_requests}")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", use_container_width=True):
        logout_user()


def enforce_session_limits(messages: list) -> bool:
    """Enforce session length limits for the current user."""
    session = get_current_session()
    if not session:
        return False
    
    if len(messages) >= session.permissions.max_session_length:
        st.warning(f"⚠️ Session limit reached! Maximum {session.permissions.max_session_length} messages per session for your role ({session.role}).")
        st.info("Start a new session to continue chatting.")
        return False
    
    return True
