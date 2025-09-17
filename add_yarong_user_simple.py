#!/usr/bin/env python3
"""
Simple script to add yaronG user when dependencies are available.
This script manually creates the user entry without requiring full dependency installation.
"""

import json
import hashlib
import secrets
import uuid
from datetime import datetime
from pathlib import Path


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{password_hash.hex()}"


def create_yarong_user_data():
    """Create yaronG user data structure."""
    
    user_data = {
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
        "created_at": datetime.utcnow().isoformat(),
        "last_login": None,
        "login_attempts": 0,
        "locked_until": None
    }
    
    return user_data


def main():
    """Create the yaronG user data file."""
    print("👑 Creating yaronG user data...")
    
    # Create the .auth directory if it doesn't exist
    auth_dir = Path(".auth")
    auth_dir.mkdir(exist_ok=True)
    
    users_file = auth_dir / "users.json"
    
    # Load existing users or create empty list
    users = []
    if users_file.exists():
        try:
            with open(users_file, 'r') as f:
                users = json.load(f)
        except:
            users = []
    
    # Check if yaronG already exists
    yarong_exists = any(user.get('username', '').lower() == 'yarong' for user in users)
    
    if yarong_exists:
        print("⚠️  yaronG user already exists!")
        return
    
    # Add yaronG user
    yarong_user = create_yarong_user_data()
    users.append(yarong_user)
    
    # Save users file
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=2)
    
    print("✅ yaronG user created successfully!")
    print(f"   Username: yaronG")
    print(f"   Email: yaron@glassboxdigital.com")
    print(f"   Role: ADMIN")
    print(f"   User ID: {yarong_user['id']}")
    print()
    print("👑 Special Features:")
    print("   - Personalized welcome: 'It's nice having you here, captain G'")
    print("   - Kingdom message: 'Welcome to your AI Agents Kingdom!'")
    print("   - Full admin privileges enabled")
    print()
    print("🚀 yaronG can now login and rule the AI Agents Playground!")


if __name__ == "__main__":
    main()
