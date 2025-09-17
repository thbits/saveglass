"""
User authentication and management module for SaveGlass.
"""
import hashlib
import json
import os
from typing import Dict, Optional, List
from pathlib import Path
from datetime import datetime
import streamlit as st
from agents_playground.logger import agent_logger


class UserManager:
    """Handle user authentication and management."""
    
    def __init__(self, storage_dir: str = None):
        """Initialize user manager with specified directory."""
        if storage_dir is None:
            # Use a .users directory in the project root
            project_root = Path(__file__).parent.parent.parent
            storage_dir = project_root / ".users"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.users_file = self.storage_dir / "users.json"
        
        # Create default admin user if no users exist
        self._ensure_default_admin()
    
    def _hash_password(self, password: str) -> str:
        """Hash password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _ensure_default_admin(self):
        """Ensure a default admin user exists."""
        users = self._load_users()
        if not users:
            # Create default admin user
            default_admin = {
                "username": "yaronG",
                "password_hash": self._hash_password("ysyghb"),
                "role": "admin",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
            users["yaronG"] = default_admin
            self._save_users(users)
            agent_logger.info("Created default admin user (username: yaronG, password: ysyghb)")
    
    def _load_users(self) -> Dict[str, Dict]:
        """Load users from storage."""
        try:
            if not self.users_file.exists():
                return {}
            
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_storage_load"})
            return {}
    
    def _save_users(self, users: Dict[str, Dict]) -> bool:
        """Save users to storage."""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2, ensure_ascii=False)
            return True
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_storage_save"})
            return False
    
    def authenticate(self, username: str, password: str) -> bool:
        """Authenticate user with username and password."""
        try:
            users = self._load_users()
            
            if username not in users:
                agent_logger.info(f"Authentication failed: user '{username}' not found")
                return False
            
            user = users[username]
            password_hash = self._hash_password(password)
            
            if user["password_hash"] == password_hash:
                # Update last login
                user["last_login"] = datetime.now().isoformat()
                users[username] = user
                self._save_users(users)
                
                agent_logger.info(f"User '{username}' authenticated successfully")
                return True
            else:
                agent_logger.info(f"Authentication failed: incorrect password for user '{username}'")
                return False
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_authentication"})
            return False
    
    def create_user(self, username: str, password: str, role: str = "user") -> bool:
        """Create a new user."""
        try:
            users = self._load_users()
            
            if username in users:
                agent_logger.info(f"User creation failed: username '{username}' already exists")
                return False
            
            new_user = {
                "username": username,
                "password_hash": self._hash_password(password),
                "role": role,
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
            
            users[username] = new_user
            success = self._save_users(users)
            
            if success:
                agent_logger.info(f"User '{username}' created successfully")
            
            return success
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_creation"})
            return False
    
    def get_user(self, username: str) -> Optional[Dict]:
        """Get user information (without password hash)."""
        try:
            users = self._load_users()
            
            if username not in users:
                return None
            
            user = users[username].copy()
            # Remove password hash for security
            user.pop("password_hash", None)
            return user
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_get"})
            return None
    
    def list_users(self) -> List[Dict]:
        """List all users (without password hashes)."""
        try:
            users = self._load_users()
            user_list = []
            
            for username, user_data in users.items():
                user = user_data.copy()
                user.pop("password_hash", None)  # Remove password hash
                user_list.append(user)
            
            return user_list
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_list"})
            return []
    
    def change_password(self, username: str, old_password: str, new_password: str) -> bool:
        """Change user password."""
        try:
            users = self._load_users()
            
            if username not in users:
                return False
            
            # Verify old password
            old_password_hash = self._hash_password(old_password)
            if users[username]["password_hash"] != old_password_hash:
                agent_logger.info(f"Password change failed: incorrect old password for user '{username}'")
                return False
            
            # Update password
            users[username]["password_hash"] = self._hash_password(new_password)
            success = self._save_users(users)
            
            if success:
                agent_logger.info(f"Password changed successfully for user '{username}'")
            
            return success
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "password_change"})
            return False
    
    def delete_user(self, username: str) -> bool:
        """Delete a user."""
        try:
            users = self._load_users()
            
            if username not in users:
                return False
            
            # Don't allow deleting the last admin user
            admin_count = sum(1 for user in users.values() if user.get("role") == "admin")
            if users[username].get("role") == "admin" and admin_count <= 1:
                agent_logger.info(f"Cannot delete user '{username}': last admin user")
                return False
            
            del users[username]
            success = self._save_users(users)
            
            if success:
                agent_logger.info(f"User '{username}' deleted successfully")
            
            return success
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "user_deletion"})
            return False


def show_login_page():
    """Display the login page."""
    st.set_page_config(
        page_title="SaveGlass - Login",
        page_icon="🔍",
        layout="centered"
    )
    
    # Custom CSS for login page
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    .login-title {
        text-align: center;
        color: #251B9C;
        margin-bottom: 2rem;
        font-size: 2.5rem;
        font-weight: 600;
    }
    .login-form {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .stTextInput > div > div > input {
        border-radius: 5px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #251B9C;
        outline: none;
    }
    .login-button {
        background-color: #251B9C;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        margin-top: 1rem;
    }
    .login-button:hover {
        background-color: #1e1575;
    }
    .error-message {
        color: #d32f2f;
        text-align: center;
        margin-top: 1rem;
        font-weight: 500;
    }
    .info-message {
        color: #1976d2;
        text-align: center;
        margin-top: 1rem;
        font-size: 0.9rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Logo and title
    logo_path = os.path.join(os.path.dirname(__file__), "../../resources/ui/glassbox-logo.svg")
    if os.path.exists(logo_path):
        with open(logo_path, "r") as f:
            logo_svg = f.read()
        st.markdown(f'''
        <div style="text-align: center; margin-bottom: 2rem;">
            <div style="width: 80px; height: 80px; margin: 0 auto; margin-bottom: 1rem;">{logo_svg}</div>
            <h1 class="login-title">SaveGlass</h1>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown('<h1 class="login-title">🔍 SaveGlass</h1>', unsafe_allow_html=True)
    
    # Login form
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        with st.form("login_form", clear_on_submit=False):
            st.markdown("### Please log in to continue")
            
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            login_button = st.form_submit_button("Log In", use_container_width=True)
            
            if login_button:
                if username and password:
                    user_manager = UserManager()
                    
                    if user_manager.authenticate(username, password):
                        # Set authentication state
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.session_state.user_info = user_manager.get_user(username)
                        
                        st.success("Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        # Default credentials info
        st.markdown("""
        <div class="info-message">
            <strong>Default Login:</strong><br>
            Username: yaronG<br>
            Password: ysyghb
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)


def check_authentication():
    """Check if user is authenticated."""
    return st.session_state.get("authenticated", False)


def logout():
    """Log out the current user."""
    if "authenticated" in st.session_state:
        username = st.session_state.get("username", "unknown")
        agent_logger.info(f"User '{username}' logged out")
        
        # Clear authentication state
        del st.session_state.authenticated
        del st.session_state.username
        if "user_info" in st.session_state:
            del st.session_state.user_info
        
        # Clear other session data
        if "messages" in st.session_state:
            del st.session_state.messages
        if "chat_sessions" in st.session_state:
            del st.session_state.chat_sessions
        if "current_session_id" in st.session_state:
            del st.session_state.current_session_id
        if "agent" in st.session_state:
            del st.session_state.agent
        
        st.rerun()


def get_current_user() -> Optional[str]:
    """Get the current authenticated username."""
    return st.session_state.get("username")


def get_user_info() -> Optional[Dict]:
    """Get current user information."""
    return st.session_state.get("user_info")


# Global user manager instance
user_manager = UserManager()
