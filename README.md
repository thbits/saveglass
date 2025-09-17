# AI Agents Playground

A modern AI agents platform with a customizable chat interface, AWS Bedrock integration, and advanced visualization capabilities.

## Features

- 🤖 **AI Chat Interface**: Interactive chat powered by AWS Bedrock
- 📊 **Dynamic Visualizations**: Automatic chart and graph generation
- ⚙️ **Configurable Agents**: Customizable AI model parameters
- 🐳 **Docker Support**: Easy deployment with hot reloading
- 🎨 **Customizable UI**: Modern Streamlit interface
- 📈 **Multiple Chart Types**: Bar, line, scatter, pie, and histogram charts
- 🔐 **User Authentication**: Secure login system with role-based permissions
- 👥 **User Management**: Multi-level access control (Admin, Power User, User, Guest)
- 🛠️ **Admin Panel**: Complete user and session management interface
- 🔒 **Permission-Based Access**: Feature access based on user roles and permissions

## Quick Start

### Prerequisites

- Docker and Docker Compose
- AWS Account with Bedrock access (optional for demo mode)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd agents-playground
```

### 2. Configure Environment (Optional)

Create a `.env` file for AWS configuration:

```bash
cp .env.example .env
# Edit .env with your AWS credentials
```

### 3. Run with Docker

```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

### 4. Access the Application

Open your browser and navigate to: `http://localhost:8501`

### 5. Login with Default Admin Account

**Default Credentials:**
- Username: `admin`
- Password: `admin123`

⚠️ **Important:** Change the default admin password immediately in production!

## Configuration

### AWS Bedrock Setup

To use AWS Bedrock models, configure these environment variables:

```bash
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

**Note**: The application works in demo mode without AWS credentials.

### Model Configuration

Available models:
- `anthropic.claude-3-sonnet-20240229-v1:0` (default)
- `anthropic.claude-3-haiku-20240307-v1:0`

Adjust parameters in the sidebar:
- **Temperature**: Controls randomness (0.0-1.0)
- **Max Tokens**: Maximum response length

## Usage

### Basic Chat

Simply type your questions in the chat input. The AI will respond contextually.

### Generating Visualizations

Request charts using keywords like:
- "Generate a sales chart"
- "Create a scatter plot"
- "Show me a bar graph"
- "Make a pie chart for demographics"

### Available Chart Types

1. **Bar Charts**: For categorical comparisons
2. **Line Charts**: For time series and trends
3. **Scatter Plots**: For correlation analysis
4. **Pie Charts**: For proportional data
5. **Histograms**: For distribution analysis

## User Management

### User Roles & Permissions

The application supports four user roles with different permission levels:

#### **Admin**
- ✅ Full access to all features
- ✅ Can access AWS cost analysis tools
- ✅ Can change LLM configurations
- ✅ Can view system information and logs
- ✅ Can export chat sessions
- ✅ Can manage other users
- ✅ 50 messages per session, 1000 daily requests
- ✅ Access to all prompt types

#### **Power User**
- ✅ Can access AWS cost analysis tools
- ✅ Can change LLM configurations
- ✅ Can export chat sessions
- ❌ Cannot manage users or view system info
- ✅ 30 messages per session, 500 daily requests
- ✅ Access to basic, advanced, and AWS analysis prompts

#### **User**
- ❌ Limited access to basic chat functionality
- ❌ Cannot access AWS tools or change configurations
- ❌ Cannot export sessions or view system info
- ✅ 20 messages per session, 200 daily requests
- ✅ Access to basic prompts only

#### **Guest**
- ❌ Very limited access
- ❌ Basic chat only with strict limits
- ✅ 5 messages per session, 20 daily requests
- ✅ Access to basic prompts only

### Admin Panel

Admins can access a comprehensive admin panel to:

- **Manage Users**: View, create, activate/deactivate, and change user roles
- **Monitor Sessions**: View active sessions and cleanup expired ones
- **System Information**: View user statistics, role distribution, and storage info
- **User Activity**: Monitor recent login activity and user behavior

To access the admin panel:
```bash
streamlit run src/agents_playground/admin_panel.py --server.port 8502
```

Then visit: `http://localhost:8502`

### Authentication Features

- **Secure Password Hashing**: Uses PBKDF2 with SHA-256 and salt
- **Session Management**: Automatic session timeout and cleanup
- **Account Lockout**: Protection against brute force attacks
- **Permission Validation**: Real-time checking of user permissions
- **Audit Logging**: Comprehensive logging of authentication events

## Development

### Hot Reloading

The Docker setup supports hot reloading. Changes to source code are automatically reflected:

```bash
# Start development environment
docker-compose up

# Edit files in src/agents_playground/
# Changes are automatically detected and applied
```

### Local Development

For local development without Docker:

```bash
# Install dependencies
pip install -e .

# Run the application
streamlit run src/agents_playground/app.py
```

### Project Structure

```
agents-playground/
├── src/agents_playground/
│   ├── __init__.py
│   ├── app.py              # Main Streamlit application
│   ├── agent.py            # AI agent implementation
│   └── utils.py            # Plotting and data utilities
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container setup
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project metadata
└── README.md              # This file
```

### Adding New Features

1. **New Chart Types**: Add functions to `src/agents_playground/utils.py`
2. **Agent Enhancements**: Modify `src/agents_playground/agent.py`
3. **UI Changes**: Update `src/agents_playground/app.py`

## Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose logs

# Rebuild container
docker-compose down
docker-compose up --build
```

**AWS Connection Issues:**
- Verify AWS credentials in environment variables
- Check AWS region configuration
- Ensure Bedrock access is enabled for your AWS account

**Port Already in Use:**
```bash
# Change port in docker-compose.yml
ports:
  - "8502:8501"  # Use different host port
```

### Health Checks

The application includes health checks:
```bash
# Check container health
docker-compose ps

# Manual health check
curl http://localhost:8501/_stcore/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with Docker
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review container logs
3. Open an issue on the repository

---

Built with ❤️ using Streamlit, LangChain, and AWS Bedrock.