#!/usr/bin/env python3
"""
Script to create yaronG user - The king of Glassbox
"""

import sys
from pathlib import Path

# Add the src directory to the path
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from agents_playground.auth_manager import auth_manager
from agents_playground.models import UserCreate, UserRole


def create_yaron_user():
    """Create yaronG user with admin role."""
    
    print("👑 Creating yaronG - The king of Glassbox...")
    print("=" * 50)
    
    try:
        user_create = UserCreate(
            username="yaronG",
            email="yaron@glassboxdigital.com",
            password="ysyghb",
            role=UserRole.ADMIN
        )
        
        result = auth_manager.create_user(user_create)
        
        if result.success:
            print("✅ Successfully created yaronG user!")
            print(f"   Username: yaronG")
            print(f"   Email: yaron@glassboxdigital.com")
            print(f"   Role: ADMIN")
            print(f"   Description: The king of Glassbox")
            print()
            print("👑 Special welcome message configured!")
            print("   When yaronG logs in, they will see:")
            print("   'It's nice having you here, captain G'")
            print()
            print("🎉 yaronG is ready to rule the AI Agents Playground!")
            
        else:
            if "already exists" in result.message:
                print("⚠️  yaronG user already exists!")
                print("   The user is ready to login with the configured credentials.")
            else:
                print(f"❌ Failed to create yaronG: {result.message}")
                
    except Exception as e:
        print(f"❌ Error creating yaronG: {str(e)}")


if __name__ == "__main__":
    create_yaron_user()
