#!/usr/bin/env python3
"""
Demo script to create sample users for testing the authentication system.
Run this script to populate the system with demo users for each role.
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from agents_playground.auth_manager import auth_manager
from agents_playground.models import UserCreate, UserRole


def create_demo_users():
    """Create demo users for testing."""
    
    demo_users = [
        {
            "username": "demo_admin",
            "email": "admin@demo.com",
            "password": "admin_demo_123",
            "role": UserRole.ADMIN,
            "description": "Demo admin user with full permissions"
        },
        {
            "username": "demo_power",
            "email": "power@demo.com", 
            "password": "power_demo_123",
            "role": UserRole.POWER_USER,
            "description": "Demo power user with AWS access"
        },
        {
            "username": "demo_user",
            "email": "user@demo.com",
            "password": "user_demo_123", 
            "role": UserRole.USER,
            "description": "Demo regular user with basic access"
        },
        {
            "username": "demo_guest",
            "email": "guest@demo.com",
            "password": "guest_demo_123",
            "role": UserRole.GUEST,
            "description": "Demo guest user with limited access"
        }
    ]
    
    print("🔧 Creating demo users for AI Agents Playground...")
    print("=" * 60)
    
    for user_info in demo_users:
        try:
            user_create = UserCreate(
                username=user_info["username"],
                email=user_info["email"],
                password=user_info["password"],
                role=user_info["role"]
            )
            
            result = auth_manager.create_user(user_create)
            
            if result.success:
                print(f"✅ Created {user_info['role'].value}: {user_info['username']}")
                print(f"   Email: {user_info['email']}")
                print(f"   Password: {user_info['password']}")
                print(f"   Description: {user_info['description']}")
                print()
            else:
                if "already exists" in result.message:
                    print(f"⚠️  {user_info['username']} already exists - skipping")
                else:
                    print(f"❌ Failed to create {user_info['username']}: {result.message}")
                print()
                
        except Exception as e:
            print(f"❌ Error creating {user_info['username']}: {str(e)}")
            print()
    
    print("=" * 60)
    print("🎉 Demo user creation completed!")
    print()
    print("📋 Summary of Demo Accounts:")
    print("┌─────────────┬─────────────────┬─────────────────┬─────────────┐")
    print("│ Username    │ Password        │ Email           │ Role        │")
    print("├─────────────┼─────────────────┼─────────────────┼─────────────┤")
    print("│ admin       │ admin123        │ admin@localhost │ admin       │")
    print("│ demo_admin  │ admin_demo_123  │ admin@demo.com  │ admin       │")
    print("│ demo_power  │ power_demo_123  │ power@demo.com  │ power_user  │")
    print("│ demo_user   │ user_demo_123   │ user@demo.com   │ user        │")
    print("│ demo_guest  │ guest_demo_123  │ guest@demo.com  │ guest       │")
    print("└─────────────┴─────────────────┴─────────────────┴─────────────┘")
    print()
    print("🚀 You can now test different user roles and permissions!")
    print("🔐 Login at: http://localhost:8501")
    print("🛠️  Admin panel at: http://localhost:8502 (for admin users)")


if __name__ == "__main__":
    create_demo_users()
