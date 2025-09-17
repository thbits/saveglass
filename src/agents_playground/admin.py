"""
Admin interface for user management in SaveGlass.
"""
import streamlit as st
from typing import List, Dict
from agents_playground.auth import user_manager, get_current_user, get_user_info
from agents_playground.logger import agent_logger


def show_user_management():
    """Display user management interface for admins."""
    current_user_info = get_user_info()
    
    # Check if current user is admin
    if not current_user_info or current_user_info.get("role") != "admin":
        st.error("Access denied. Admin privileges required.")
        return
    
    # Add back button
    if st.button("← Back to Chat", use_container_width=True):
        st.session_state.show_admin_panel = False
        st.rerun()
    
    st.title("👥 User Management")
    st.markdown("Manage user accounts and permissions.")
    
    # Create tabs for different admin functions
    tab1, tab2, tab3 = st.tabs(["👀 View Users", "➕ Add User", "🔧 Manage Users"])
    
    with tab1:
        st.subheader("Current Users")
        users = user_manager.list_users()
        
        if users:
            # Create a more readable display
            for user in users:
                with st.expander(f"👤 {user['username']} ({user['role']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Username:** {user['username']}")
                        st.write(f"**Role:** {user['role']}")
                        st.write(f"**Created:** {user['created_at'][:19]}")
                    
                    with col2:
                        last_login = user.get('last_login', 'Never')
                        if last_login != 'Never':
                            last_login = last_login[:19]
                        st.write(f"**Last Login:** {last_login}")
                        
                        # Show user actions
                        if user['username'] != get_current_user():
                            if st.button(f"🗑️ Delete", key=f"delete_{user['username']}"):
                                if user_manager.delete_user(user['username']):
                                    st.success(f"User {user['username']} deleted successfully")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete user {user['username']}")
        else:
            st.info("No users found.")
    
    with tab2:
        st.subheader("Add New User")
        
        with st.form("add_user_form"):
            new_username = st.text_input("Username", placeholder="Enter username")
            new_password = st.text_input("Password", type="password", placeholder="Enter password")
            new_role = st.selectbox("Role", ["user", "admin"], index=0)
            
            submit_button = st.form_submit_button("Add User")
            
            if submit_button:
                if new_username and new_password:
                    if user_manager.create_user(new_username, new_password, new_role):
                        st.success(f"User '{new_username}' created successfully!")
                        agent_logger.info(f"New user created by admin: {new_username} (role: {new_role})")
                        st.rerun()
                    else:
                        st.error("Failed to create user. Username may already exist.")
                else:
                    st.error("Please provide both username and password.")
    
    with tab3:
        st.subheader("User Management Tools")
        
        # Password reset section
        st.markdown("#### Reset User Password")
        users = user_manager.list_users()
        user_list = [user['username'] for user in users if user['username'] != get_current_user()]
        
        if user_list:
            with st.form("reset_password_form"):
                selected_user = st.selectbox("Select User", user_list)
                new_password = st.text_input("New Password", type="password", placeholder="Enter new password")
                
                reset_button = st.form_submit_button("Reset Password")
                
                if reset_button and new_password:
                    # For admin reset, we'll directly update the password
                    users_data = user_manager._load_users()
                    if selected_user in users_data:
                        users_data[selected_user]['password_hash'] = user_manager._hash_password(new_password)
                        if user_manager._save_users(users_data):
                            st.success(f"Password reset for user '{selected_user}'")
                            agent_logger.info(f"Password reset by admin for user: {selected_user}")
                        else:
                            st.error("Failed to reset password")
                    else:
                        st.error("User not found")
        else:
            st.info("No other users to manage.")
        
        # System information
        st.markdown("#### System Information")
        st.info(f"Total Users: {len(users)}")
        admin_count = sum(1 for user in users if user.get('role') == 'admin')
        st.info(f"Admin Users: {admin_count}")


def show_admin_menu():
    """Show admin menu in sidebar if user is admin."""
    current_user_info = get_user_info()
    
    if current_user_info and current_user_info.get("role") == "admin":
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Admin Panel")
        
        if st.sidebar.button("👥 User Management", use_container_width=True):
            st.session_state.show_admin_panel = True
            st.rerun()
