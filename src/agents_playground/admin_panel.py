"""
Admin panel for user management in AI Agents Playground.
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from agents_playground.auth_manager import auth_manager
from agents_playground.auth_utils import get_current_session, require_role
from agents_playground.models import UserCreate, UserRole, UserPermissions
from agents_playground.logger import agent_logger


@require_role(UserRole.ADMIN)
def show_admin_panel():
    """Display the admin panel for user management."""
    st.title("🛠️ Admin Panel")
    st.markdown("Manage users, permissions, and system settings.")
    
    tab1, tab2, tab3, tab4 = st.tabs(["👥 Users", "➕ Create User", "📊 Sessions", "🔧 System"])
    
    with tab1:
        show_users_management()
    
    with tab2:
        show_create_user()
    
    with tab3:
        show_sessions_management()
    
    with tab4:
        show_system_info()


def show_users_management():
    """Show user management interface."""
    st.subheader("👥 User Management")
    
    try:
        users = auth_manager._load_users()
        
        if not users:
            st.info("No users found.")
            return
        
        # Create users DataFrame
        users_data = []
        for user in users:
            users_data.append({
                "Username": user.username,
                "Email": user.email,
                "Role": user.role,
                "Active": "✅" if user.is_active else "❌",
                "Created": user.created_at.strftime("%Y-%m-%d %H:%M") if user.created_at else "Unknown",
                "Last Login": user.last_login.strftime("%Y-%m-%d %H:%M") if user.last_login else "Never",
                "Login Attempts": user.login_attempts,
                "AWS Tools": "✅" if user.permissions.can_access_aws_tools else "❌",
                "LLM Config": "✅" if user.permissions.can_change_llm_config else "❌",
                "Max Sessions": user.permissions.max_session_length,
                "Daily Requests": user.permissions.max_daily_requests
            })
        
        df = pd.DataFrame(users_data)
        st.dataframe(df, use_container_width=True)
        
        # User actions
        st.subheader("🔧 User Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Modify User Status:**")
            selected_user = st.selectbox(
                "Select User",
                [user.username for user in users],
                key="modify_user_select"
            )
            
            if selected_user:
                user = next((u for u in users if u.username == selected_user), None)
                if user:
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button(f"{'Deactivate' if user.is_active else 'Activate'} User"):
                            user.is_active = not user.is_active
                            users = [u if u.username != selected_user else user for u in users]
                            if auth_manager._save_users(users):
                                st.success(f"User {'activated' if user.is_active else 'deactivated'} successfully!")
                                st.rerun()
                    
                    with col_b:
                        if user.login_attempts > 0:
                            if st.button("Reset Login Attempts"):
                                user.login_attempts = 0
                                user.locked_until = None
                                users = [u if u.username != selected_user else user for u in users]
                                if auth_manager._save_users(users):
                                    st.success("Login attempts reset successfully!")
                                    st.rerun()
        
        with col2:
            st.markdown("**Change User Role:**")
            role_user = st.selectbox(
                "Select User for Role Change",
                [user.username for user in users],
                key="role_user_select"
            )
            
            if role_user:
                user = next((u for u in users if u.username == role_user), None)
                if user:
                    new_role = st.selectbox(
                        "New Role",
                        [role.value for role in UserRole],
                        index=[role.value for role in UserRole].index(user.role.value),
                        key="new_role_select"
                    )
                    
                    if st.button("Update Role"):
                        user.role = UserRole(new_role)
                        user.permissions = auth_manager._get_role_permissions(user.role)
                        users = [u if u.username != role_user else user for u in users]
                        if auth_manager._save_users(users):
                            st.success(f"Role updated to {new_role} successfully!")
                            st.rerun()
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "admin_users_management"})
        st.error(f"Error loading users: {str(e)}")


def show_create_user():
    """Show create user interface."""
    st.subheader("➕ Create New User")
    
    with st.form("create_user_form"):
        st.markdown("### User Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            username = st.text_input("Username*", placeholder="Enter username (3-50 characters)")
            email = st.text_input("Email*", placeholder="Enter email address")
        
        with col2:
            password = st.text_input("Password*", type="password", placeholder="Enter password (min 8 characters)")
            confirm_password = st.text_input("Confirm Password*", type="password", placeholder="Confirm password")
        
        role = st.selectbox(
            "User Role*",
            [role.value for role in UserRole],
            index=2,  # Default to USER role
            help="Select the user's role and permission level"
        )
        
        # Show role permissions preview
        preview_role = UserRole(role)
        preview_permissions = auth_manager._get_role_permissions(preview_role)
        
        with st.expander("🔍 Role Permissions Preview"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write(f"**AWS Tools:** {'✅' if preview_permissions.can_access_aws_tools else '❌'}")
                st.write(f"**LLM Config:** {'✅' if preview_permissions.can_change_llm_config else '❌'}")
                st.write(f"**System Info:** {'✅' if preview_permissions.can_view_system_info else '❌'}")
            with col_b:
                st.write(f"**Export Sessions:** {'✅' if preview_permissions.can_export_sessions else '❌'}")
                st.write(f"**Max Session Length:** {preview_permissions.max_session_length}")
                st.write(f"**Daily Requests:** {preview_permissions.max_daily_requests}")
        
        create_button = st.form_submit_button("Create User", use_container_width=True)
        
        if create_button:
            # Validation
            errors = []
            
            if not username or len(username) < 3 or len(username) > 50:
                errors.append("Username must be 3-50 characters long")
            
            if not email or "@" not in email:
                errors.append("Please enter a valid email address")
            
            if not password or len(password) < 8:
                errors.append("Password must be at least 8 characters long")
            
            if password != confirm_password:
                errors.append("Passwords do not match")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                try:
                    user_create = UserCreate(
                        username=username,
                        email=email,
                        password=password,
                        role=UserRole(role)
                    )
                    
                    result = auth_manager.create_user(user_create)
                    
                    if result.success:
                        st.success(f"User '{username}' created successfully!")
                        agent_logger.info(f"Admin created new user: {username} with role {role}")
                        st.rerun()
                    else:
                        st.error(f"Failed to create user: {result.message}")
                        
                except Exception as e:
                    agent_logger.log_error(e, {"context": "admin_create_user"})
                    st.error(f"Error creating user: {str(e)}")


def show_sessions_management():
    """Show session management interface."""
    st.subheader("📊 Active Sessions")
    
    try:
        sessions = auth_manager._load_sessions()
        active_sessions = [s for s in sessions if s.expires_at > datetime.utcnow()]
        
        if not active_sessions:
            st.info("No active sessions found.")
            return
        
        # Create sessions DataFrame
        sessions_data = []
        for session in active_sessions:
            time_remaining = session.expires_at - datetime.utcnow()
            hours_remaining = time_remaining.total_seconds() / 3600
            
            sessions_data.append({
                "Username": session.username,
                "Role": session.role,
                "Session ID": session.session_id[:8] + "...",
                "Created": session.created_at.strftime("%Y-%m-%d %H:%M"),
                "Last Activity": session.last_activity.strftime("%Y-%m-%d %H:%M"),
                "Expires In": f"{hours_remaining:.1f}h",
                "AWS Access": "✅" if session.permissions.can_access_aws_tools else "❌",
                "LLM Config": "✅" if session.permissions.can_change_llm_config else "❌"
            })
        
        df = pd.DataFrame(sessions_data)
        st.dataframe(df, use_container_width=True)
        
        # Session cleanup
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Cleanup Expired Sessions"):
                auth_manager.cleanup_expired_sessions()
                st.success("Expired sessions cleaned up!")
                st.rerun()
        
        with col2:
            if st.button("📊 Refresh Sessions"):
                st.rerun()
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "admin_sessions_management"})
        st.error(f"Error loading sessions: {str(e)}")


def show_system_info():
    """Show system information."""
    st.subheader("🔧 System Information")
    
    try:
        # User statistics
        users = auth_manager._load_users()
        sessions = auth_manager._load_sessions()
        active_sessions = [s for s in sessions if s.expires_at > datetime.utcnow()]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Users", len(users))
        
        with col2:
            active_users = len([u for u in users if u.is_active])
            st.metric("Active Users", active_users)
        
        with col3:
            st.metric("Active Sessions", len(active_sessions))
        
        with col4:
            admin_count = len([u for u in users if u.role == UserRole.ADMIN])
            st.metric("Admin Users", admin_count)
        
        # Role distribution
        st.subheader("👥 User Role Distribution")
        role_counts = {}
        for user in users:
            role_counts[user.role.value] = role_counts.get(user.role.value, 0) + 1
        
        role_df = pd.DataFrame(list(role_counts.items()), columns=["Role", "Count"])
        st.bar_chart(role_df.set_index("Role"))
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        recent_logins = []
        for user in users:
            if user.last_login:
                recent_logins.append({
                    "Username": user.username,
                    "Role": user.role.value,
                    "Last Login": user.last_login.strftime("%Y-%m-%d %H:%M"),
                    "Days Ago": (datetime.utcnow() - user.last_login).days
                })
        
        if recent_logins:
            recent_df = pd.DataFrame(recent_logins)
            recent_df = recent_df.sort_values("Last Login", ascending=False).head(10)
            st.dataframe(recent_df, use_container_width=True)
        else:
            st.info("No recent login activity.")
        
        # Storage info
        st.subheader("💾 Storage Information")
        
        # Calculate storage sizes
        import os
        
        auth_dir = auth_manager.storage_dir
        users_file_size = 0
        sessions_file_size = 0
        
        if auth_manager.users_file.exists():
            users_file_size = auth_manager.users_file.stat().st_size
        
        if auth_manager.sessions_file.exists():
            sessions_file_size = auth_manager.sessions_file.stat().st_size
        
        storage_col1, storage_col2 = st.columns(2)
        
        with storage_col1:
            st.metric("Users File Size", f"{users_file_size / 1024:.1f} KB")
            st.metric("Sessions File Size", f"{sessions_file_size / 1024:.1f} KB")
        
        with storage_col2:
            st.metric("Storage Directory", str(auth_dir))
            total_size = users_file_size + sessions_file_size
            st.metric("Total Auth Storage", f"{total_size / 1024:.1f} KB")
        
    except Exception as e:
        agent_logger.log_error(e, {"context": "admin_system_info"})
        st.error(f"Error loading system information: {str(e)}")


if __name__ == "__main__":
    # This allows the admin panel to be run as a standalone page
    show_admin_panel()
