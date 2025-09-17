# User Management and Authentication

SaveGlass now includes a comprehensive user management system with login authentication and admin capabilities.

## Features

### 🔐 User Authentication
- **Login Page**: Secure username/password authentication
- **Session Management**: User-specific chat sessions and data isolation
- **Automatic Logout**: Session expiration and manual logout functionality

### 👥 User Roles
- **Admin**: Full system access with user management capabilities
- **User**: Standard chat functionality with personal session storage

### 🛡️ Security Features
- **Password Hashing**: SHA-256 password encryption
- **Session Isolation**: Each user's chat sessions are stored separately
- **Access Control**: Role-based access to admin functions

## Getting Started

### Default Credentials
When you first run the application, a default admin account is automatically created:

- **Username**: `yaronG`
- **Password**: `ysyghb`

⚠️ **Security Note**: Please change the default admin password immediately after first login!

### Login Process
1. Navigate to the SaveGlass application
2. You'll be automatically redirected to the login page
3. Enter your username and password
4. Click "Log In" to access the application

### User Interface
After logging in, you'll see:
- **Welcome Message**: Your username displayed in the sidebar
- **Logout Button**: Easily sign out of your session
- **Personal Chat Sessions**: Your own chat history isolated from other users
- **Admin Panel** (Admin users only): Access to user management

## Admin Features

Admin users have access to additional functionality:

### User Management Panel
Access via the "👥 User Management" button in the sidebar.

#### View Users Tab
- See all registered users
- View user details (username, role, creation date, last login)
- Delete users (except the last admin)

#### Add User Tab
- Create new user accounts
- Assign user or admin roles
- Set initial passwords

#### Manage Users Tab
- Reset user passwords
- View system statistics
- Monitor user activity

## File Structure

The authentication system consists of several new files:

```
src/agents_playground/
├── auth.py           # Core authentication functions
├── admin.py          # Admin interface and user management
├── app.py            # Updated main app with auth integration
└── session_storage.py # Updated with user-specific storage
```

### Data Storage

User data and sessions are stored in:
```
.users/
├── users.json        # User accounts and credentials
.sessions/
├── chat_sessions_[username].json  # User-specific chat sessions
```

## API Reference

### Authentication Functions

```python
from agents_playground.auth import (
    check_authentication,  # Check if user is logged in
    get_current_user,      # Get current username
    get_user_info,         # Get current user details
    logout,                # Log out current user
    user_manager           # Access user management functions
)
```

### User Management

```python
# Create a new user
user_manager.create_user("username", "password", "role")

# Authenticate user
user_manager.authenticate("username", "password")

# Change password
user_manager.change_password("username", "old_pass", "new_pass")

# Delete user
user_manager.delete_user("username")
```

## Security Considerations

1. **Password Security**: Passwords are hashed using SHA-256
2. **Session Isolation**: Each user's data is completely separate
3. **Admin Protection**: Cannot delete the last admin user
4. **Access Control**: Admin functions are role-protected

## Development Notes

### Session Storage Integration
The session storage system has been updated to support user-specific storage:
- Chat sessions are now stored per user
- Backward compatibility maintained for existing sessions
- User-specific file naming: `chat_sessions_[username].json`

### Authentication Flow
1. **Initial Load**: Check authentication state
2. **Login Required**: Redirect to login page if not authenticated
3. **Session Creation**: Initialize user-specific data after login
4. **Logout**: Clear all session data and redirect to login

## Troubleshooting

### Cannot Access Admin Panel
- Ensure you're logged in as an admin user
- Default admin: username `yaronG`, password `ysyghb`

### Lost Admin Access
- Check the `.users/users.json` file
- Manually edit user role to "admin" if needed
- Restart the application

### Session Data Issues
- Check `.sessions/` directory for user-specific files
- Sessions are automatically created for new users
- Clear sessions by deleting the user's session file

## Future Enhancements

Potential improvements for the authentication system:
- Email-based user registration
- Password complexity requirements
- Session timeout configuration
- User activity logging
- Password reset functionality
- OAuth integration (Google, GitHub, etc.)
