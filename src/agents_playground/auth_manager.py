"""
Authentication and user management system for AI Agents Playground.
"""
import hashlib
import json
import os
import secrets
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agents_playground.logger import agent_logger
from agents_playground.models import (
    AuthResponse, AuthSession, User, UserCreate, UserLogin, 
    UserPermissions, UserRole
)


class AuthManager:
    """Handle user authentication and session management."""
    
    def __init__(self, storage_dir: str = None):
        """Initialize authentication manager with specified directory."""
        if storage_dir is None:
            # Use a .auth directory in the project root
            project_root = Path(__file__).parent.parent.parent
            storage_dir = project_root / ".auth"
        
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.users_file = self.storage_dir / "users.json"
        self.sessions_file = self.storage_dir / "sessions.json"
        
        # Session configuration
        self.session_lifetime = timedelta(hours=24)
        self.max_login_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
        
        # Initialize with default admin user if no users exist
        self._ensure_default_admin()
    
    def _ensure_default_admin(self):
        """Ensure a default admin user exists."""
        try:
            users = self._load_users()
            if not users:
                # Create default admin user
                admin_user = UserCreate(
                    username="admin",
                    email="admin@localhost",
                    password="admin123",  # Default password - should be changed
                    role=UserRole.ADMIN
                )
                result = self.create_user(admin_user)
                if result.success:
                    agent_logger.info("Created default admin user (username: admin, password: admin123)")
                else:
                    agent_logger.error(f"Failed to create default admin user: {result.message}")
        except Exception as e:
            agent_logger.log_error(e, {"context": "ensure_default_admin"})
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 with salt."""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash."""
        try:
            salt, hash_hex = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_hash_check.hex() == hash_hex
        except (ValueError, AttributeError):
            return False
    
    def _get_role_permissions(self, role: UserRole) -> UserPermissions:
        """Get default permissions for a user role."""
        if role == UserRole.ADMIN:
            return UserPermissions(
                can_access_aws_tools=True,
                can_change_llm_config=True,
                can_view_system_info=True,
                can_export_sessions=True,
                can_manage_users=True,
                max_session_length=50,
                max_daily_requests=1000,
                allowed_prompt_types=["basic", "advanced", "system", "aws_analysis"]
            )
        elif role == UserRole.POWER_USER:
            return UserPermissions(
                can_access_aws_tools=True,
                can_change_llm_config=True,
                can_view_system_info=False,
                can_export_sessions=True,
                can_manage_users=False,
                max_session_length=30,
                max_daily_requests=500,
                allowed_prompt_types=["basic", "advanced", "aws_analysis"]
            )
        elif role == UserRole.USER:
            return UserPermissions(
                can_access_aws_tools=False,
                can_change_llm_config=False,
                can_view_system_info=False,
                can_export_sessions=False,
                can_manage_users=False,
                max_session_length=20,
                max_daily_requests=200,
                allowed_prompt_types=["basic"]
            )
        else:  # GUEST
            return UserPermissions(
                can_access_aws_tools=False,
                can_change_llm_config=False,
                can_view_system_info=False,
                can_export_sessions=False,
                can_manage_users=False,
                max_session_length=5,
                max_daily_requests=20,
                allowed_prompt_types=["basic"]
            )
    
    def _serialize_user(self, user: User) -> Dict:
        """Serialize user for JSON storage."""
        user_dict = user.model_dump()
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'last_login', 'locked_until']:
            if user_dict.get(field) and isinstance(user_dict[field], datetime):
                user_dict[field] = user_dict[field].isoformat()
        return user_dict
    
    def _deserialize_user(self, user_data: Dict) -> User:
        """Deserialize user from JSON storage."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'last_login', 'locked_until']:
            if user_data.get(field) and isinstance(user_data[field], str):
                try:
                    user_data[field] = datetime.fromisoformat(user_data[field])
                except ValueError:
                    if field == 'created_at':
                        user_data[field] = datetime.utcnow()
                    else:
                        user_data[field] = None
        return User(**user_data)
    
    def _serialize_session(self, session: AuthSession) -> Dict:
        """Serialize session for JSON storage."""
        session_dict = session.model_dump()
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'expires_at', 'last_activity']:
            if session_dict.get(field) and isinstance(session_dict[field], datetime):
                session_dict[field] = session_dict[field].isoformat()
        return session_dict
    
    def _deserialize_session(self, session_data: Dict) -> AuthSession:
        """Deserialize session from JSON storage."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'expires_at', 'last_activity']:
            if session_data.get(field) and isinstance(session_data[field], str):
                try:
                    session_data[field] = datetime.fromisoformat(session_data[field])
                except ValueError:
                    session_data[field] = datetime.utcnow()
        return AuthSession(**session_data)
    
    def _load_users(self) -> List[User]:
        """Load users from storage."""
        try:
            if not self.users_file.exists():
                return []
            
            with open(self.users_file, 'r', encoding='utf-8') as f:
                users_data = json.load(f)
            
            return [self._deserialize_user(user_data) for user_data in users_data]
        except Exception as e:
            agent_logger.log_error(e, {"context": "load_users"})
            return []
    
    def _save_users(self, users: List[User]) -> bool:
        """Save users to storage."""
        try:
            users_data = [self._serialize_user(user) for user in users]
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            agent_logger.log_error(e, {"context": "save_users"})
            return False
    
    def _load_sessions(self) -> List[AuthSession]:
        """Load sessions from storage."""
        try:
            if not self.sessions_file.exists():
                return []
            
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                sessions_data = json.load(f)
            
            return [self._deserialize_session(session_data) for session_data in sessions_data]
        except Exception as e:
            agent_logger.log_error(e, {"context": "load_sessions"})
            return []
    
    def _save_sessions(self, sessions: List[AuthSession]) -> bool:
        """Save sessions to storage."""
        try:
            sessions_data = [self._serialize_session(session) for session in sessions]
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            agent_logger.log_error(e, {"context": "save_sessions"})
            return False
    
    def create_user(self, user_create: UserCreate) -> AuthResponse:
        """Create a new user."""
        try:
            users = self._load_users()
            
            # Check if username or email already exists
            for user in users:
                if user.username.lower() == user_create.username.lower():
                    return AuthResponse(
                        success=False,
                        message="Username already exists"
                    )
                if user.email.lower() == user_create.email.lower():
                    return AuthResponse(
                        success=False,
                        message="Email already exists"
                    )
            
            # Create new user
            new_user = User(
                id=str(uuid.uuid4()),
                username=user_create.username,
                email=user_create.email,
                password_hash=self._hash_password(user_create.password),
                role=user_create.role,
                permissions=self._get_role_permissions(user_create.role),
                created_at=datetime.utcnow()
            )
            
            users.append(new_user)
            
            if self._save_users(users):
                agent_logger.info(f"Created new user: {user_create.username} with role {user_create.role}")
                return AuthResponse(
                    success=True,
                    message="User created successfully"
                )
            else:
                return AuthResponse(
                    success=False,
                    message="Failed to save user"
                )
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "create_user"})
            return AuthResponse(
                success=False,
                message=f"Error creating user: {str(e)}"
            )
    
    def authenticate_user(self, login_request: UserLogin) -> AuthResponse:
        """Authenticate a user and create a session."""
        try:
            users = self._load_users()
            user = None
            
            # Find user by username or email
            for u in users:
                if (u.username.lower() == login_request.username.lower() or 
                    u.email.lower() == login_request.username.lower()):
                    user = u
                    break
            
            if not user:
                return AuthResponse(
                    success=False,
                    message="Invalid username or password"
                )
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                return AuthResponse(
                    success=False,
                    message=f"Account is locked until {user.locked_until.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            
            # Check if account is active
            if not user.is_active:
                return AuthResponse(
                    success=False,
                    message="Account is disabled"
                )
            
            # Verify password
            if not self._verify_password(login_request.password, user.password_hash):
                # Increment login attempts
                user.login_attempts += 1
                
                # Lock account if too many attempts
                if user.login_attempts >= self.max_login_attempts:
                    user.locked_until = datetime.utcnow() + self.lockout_duration
                    agent_logger.info(f"Account locked for user {user.username} due to too many failed login attempts")
                
                # Update user in storage
                users = [u if u.id != user.id else user for u in users]
                self._save_users(users)
                
                return AuthResponse(
                    success=False,
                    message="Invalid username or password"
                )
            
            # Reset login attempts on successful password verification
            user.login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            
            # Update user in storage
            users = [u if u.id != user.id else user for u in users]
            self._save_users(users)
            
            # Create session
            session_id = str(uuid.uuid4())
            session = AuthSession(
                session_id=session_id,
                user_id=user.id,
                username=user.username,
                role=user.role,
                permissions=user.permissions,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + self.session_lifetime
            )
            
            # Save session
            sessions = self._load_sessions()
            sessions.append(session)
            
            # Clean up expired sessions
            sessions = [s for s in sessions if s.expires_at > datetime.utcnow()]
            
            if self._save_sessions(sessions):
                agent_logger.info(f"User {user.username} authenticated successfully")
                return AuthResponse(
                    success=True,
                    message="Authentication successful",
                    session_id=session_id,
                    user={
                        "id": user.id,
                        "username": user.username,
                        "email": user.email,
                        "role": user.role,
                        "last_login": user.last_login.isoformat() if user.last_login else None
                    },
                    permissions=user.permissions
                )
            else:
                return AuthResponse(
                    success=False,
                    message="Failed to create session"
                )
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "authenticate_user"})
            return AuthResponse(
                success=False,
                message=f"Authentication error: {str(e)}"
            )
    
    def validate_session(self, session_id: str) -> Optional[AuthSession]:
        """Validate and return a session if valid."""
        try:
            sessions = self._load_sessions()
            
            for session in sessions:
                if session.session_id == session_id:
                    if session.expires_at > datetime.utcnow():
                        # Update last activity
                        session.last_activity = datetime.utcnow()
                        
                        # Save updated sessions
                        sessions = [s if s.session_id != session_id else session for s in sessions]
                        self._save_sessions(sessions)
                        
                        return session
                    else:
                        # Session expired, remove it
                        sessions = [s for s in sessions if s.session_id != session_id]
                        self._save_sessions(sessions)
                        break
            
            return None
            
        except Exception as e:
            agent_logger.log_error(e, {"context": "validate_session"})
            return None
    
    def logout_user(self, session_id: str) -> AuthResponse:
        """Logout a user by removing their session."""
        try:
            sessions = self._load_sessions()
            initial_count = len(sessions)
            
            sessions = [s for s in sessions if s.session_id != session_id]
            
            if len(sessions) < initial_count:
                self._save_sessions(sessions)
                agent_logger.info(f"User logged out successfully")
                return AuthResponse(
                    success=True,
                    message="Logged out successfully"
                )
            else:
                return AuthResponse(
                    success=False,
                    message="Session not found"
                )
                
        except Exception as e:
            agent_logger.log_error(e, {"context": "logout_user"})
            return AuthResponse(
                success=False,
                message=f"Logout error: {str(e)}"
            )
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get a user by their ID."""
        try:
            users = self._load_users()
            for user in users:
                if user.id == user_id:
                    return user
            return None
        except Exception as e:
            agent_logger.log_error(e, {"context": "get_user_by_id"})
            return None
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions from storage."""
        try:
            sessions = self._load_sessions()
            active_sessions = [s for s in sessions if s.expires_at > datetime.utcnow()]
            
            if len(active_sessions) < len(sessions):
                self._save_sessions(active_sessions)
                agent_logger.info(f"Cleaned up {len(sessions) - len(active_sessions)} expired sessions")
        except Exception as e:
            agent_logger.log_error(e, {"context": "cleanup_expired_sessions"})


# Global auth manager instance
auth_manager = AuthManager()
