#!/usr/bin/env python3
"""
Startup script to ensure yaronG user exists when the application starts.
This runs automatically when the Docker container starts.
"""

import json
import hashlib
import secrets
import uuid
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, '/app/src')


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"


def ensure_yarong_user():
    """Ensure yaronG user exists in the system."""
    
    # Create auth directory if it doesn't exist
    # Try Docker path first, then local path
    if Path("/app").exists():
        auth_dir = Path("/app/.auth")
    else:
        auth_dir = Path(".auth")
    auth_dir.mkdir(exist_ok=True)
    
    users_file = auth_dir / "users.json"
    
    # Load existing users
    users = []
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing users: {e}")
            users = []
    
    # Check if yaronG already exists
    yarong_exists = any(user.get('username', '').lower() == 'yarong' for user in users)
    
    if yarong_exists:
        print("✅ yaronG user already exists")
        return
    
    # Create yaronG user
    yarong_user = {
        "id": str(uuid.uuid4()),
        "username": "yaronG",
        "email": "yaron@glassboxdigital.com",
        "password_hash": hash_password("ysyghb"),
        "role": "admin",
        "permissions": {
            "can_access_aws_tools": True,
            "can_change_llm_config": True,
            "can_view_system_info": True,
            "can_export_sessions": True,
            "can_manage_users": True,
            "max_session_length": 50,
            "max_daily_requests": 1000,
            "allowed_prompt_types": ["basic", "advanced", "system", "aws_analysis"]
        },
        "is_active": True,
        "created_at": datetime.utcnow().isoformat().replace('+00:00', ''),
        "last_login": None,
        "login_attempts": 0,
        "locked_until": None
    }
    
    # Add to users list
    users.append(yarong_user)
    
    # Save users file
    try:
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
        print("👑 yaronG user created successfully!")
        print(f"   Username: yaronG")
        print(f"   Email: yaron@glassboxdigital.com") 
        print(f"   Role: ADMIN")
        print(f"   Special welcome: 'It's nice having you here, captain G'")
    except Exception as e:
        print(f"❌ Error saving yaronG user: {e}")


def ensure_default_admin():
    """Ensure default admin user exists."""
    
    # Try Docker path first, then local path
    if Path("/app").exists():
        auth_dir = Path("/app/.auth")
    else:
        auth_dir = Path(".auth")
    auth_dir.mkdir(exist_ok=True)
    
    users_file = auth_dir / "users.json"
    
    # Load existing users
    users = []
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
        except:
            users = []
    
    # Check if admin already exists
    admin_exists = any(user.get('username', '').lower() == 'admin' for user in users)
    
    if admin_exists:
        print("✅ Default admin user already exists")
        return
    
    # Create default admin user
    admin_user = {
        "id": str(uuid.uuid4()),
        "username": "admin",
        "email": "admin@localhost",
        "password_hash": hash_password("admin123"),
        "role": "admin",
        "permissions": {
            "can_access_aws_tools": True,
            "can_change_llm_config": True,
            "can_view_system_info": True,
            "can_export_sessions": True,
            "can_manage_users": True,
            "max_session_length": 50,
            "max_daily_requests": 1000,
            "allowed_prompt_types": ["basic", "advanced", "system", "aws_analysis"]
        },
        "is_active": True,
        "created_at": datetime.utcnow().isoformat().replace('+00:00', ''),
        "last_login": None,
        "login_attempts": 0,
        "locked_until": None
    }
    
    users.append(admin_user)
    
    try:
        with open(users_file, 'w') as f:
            json.dump(users, f, indent=2)
        print("🔧 Default admin user created (username: admin, password: admin123)")
    except Exception as e:
        print(f"❌ Error saving admin user: {e}")


if __name__ == "__main__":
    print("🚀 Ensuring required users exist...")
    ensure_default_admin()
    ensure_yarong_user()
    print("✅ User setup complete!")
