# LLM Chat API - Multi-Provider POC

A production-ready FastAPI application for building LLM-powered chat applications with support for multiple providers (OpenAI & Anthropic Claude) and comprehensive performance tracking.

## ✨ Features

- 🤖 **Multi-Provider Support**: OpenAI GPT-4 and Anthropic Claude
- ⚡ **Performance Tracking**: Real-time latency and throughput metrics
- 📊 **Automatic Documentation**: Interactive Swagger UI and ReDoc
- 🚀 **Async Architecture**: Built on FastAPI for high performance
- 🔒 **Type Safety**: Pydantic models for request/response validation
- 📈 **Easy Deployment**: Ready for Render, Heroku, or any cloud platform
- 🧪 **Built-in Testing**: Performance comparison tools included

## 📋 Prerequisites

- Python 3.8+
- OpenAI API Key (optional, if using OpenAI)
- Anthropic API Key (optional, if using Claude)
- At least one API key is required

## 🚀 Quick Start

### 1. Clone or Download the Project

```bash
cd llm-chat-api
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your API keys
# nano .env  # or use any text editor
```

Edit `.env` file:
```env
LLM_PROVIDER=openai  # or anthropic

# Add your API keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model configuration
OPENAI_MODEL=gpt-4-turbo-preview
ANTHROPIC_MODEL=claude-sonnet-4-20250514
```

### 5. Run the Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📚 API Endpoints

### Health Check
```bash
GET /
```

### Get Available Providers
```bash
GET /api/v1/providers
```

### Chat (Full Featured)
```bash
POST /api/v1/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What is machine learning?"}
  ],
  "max_tokens": 500,
  "temperature": 0.7,
  "provider": "openai"
}
```

### Simple Chat
```bash
POST /api/v1/chat/simple?message=Hello&provider=openai
```

## 🧪 Testing

### Using curl

```bash
# Test OpenAI
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "provider": "openai",
    "max_tokens": 500
  }'

# Test Anthropic
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is AI?"}],
    "provider": "anthropic",
    "max_tokens": 500
  }'
```

### Using the Performance Test Script

```bash
python test_performance.py
```

This will:
- Test all available providers
- Make 5 requests to each provider
- Calculate average latency, min/max times, and tokens/second
- Display detailed performance statistics

## 📊 Response Format

```json
{
  "response": "Machine learning is...",
  "model": "gpt-4-turbo-preview",
  "provider": "openai",
  "usage": {
    "input_tokens": 15,
    "output_tokens": 106,
    "total_tokens": 121
  },
  "performance": {
    "total_time_seconds": 2.3456,
    "tokens_per_second": 45.2,
    "input_tokens": 15,
    "output_tokens": 106,
    "total_tokens": 121
  },
  "timestamp": "2024-03-29T10:30:00.000Z"
}
```

## 🚀 Deploying to Render

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/yourusername/llm-chat-api.git
git push -u origin main
```

### 2. Create Web Service on Render

1. Go to https://render.com
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: llm-chat-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Free (or paid for production)

### 3. Add Environment Variables

In Render dashboard, add:
- `OPENAI_API_KEY`: your_key_here
- `ANTHROPIC_API_KEY`: your_key_here
- `LLM_PROVIDER`: openai (or anthropic)
- `ENVIRONMENT`: production

### 4. Deploy

Click "Create Web Service" and wait for deployment!

Your API will be at: `https://your-app-name.onrender.com`

## 📁 Project Structure

```
llm-chat-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   └── llm_service.py   # LLM integration
│   └── routes/
│       ├── __init__.py
│       └── chat.py          # API endpoints
├── requirements.txt
├── .env.example
├── .gitignore
├── Procfile
├── test_performance.py
└── README.md
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | Default provider (openai/anthropic) | openai |
| `OPENAI_API_KEY` | OpenAI API key | - |
| `ANTHROPIC_API_KEY` | Anthropic API key | - |
| `OPENAI_MODEL` | OpenAI model name | gpt-4-turbo-preview |
| `ANTHROPIC_MODEL` | Anthropic model name | claude-sonnet-4-20250514 |
| `ENVIRONMENT` | Environment (development/production) | development |
| `MAX_TOKENS` | Default max tokens | 1024 |

## 🎯 Performance Metrics Explained

- **total_time_seconds**: Complete request-response time
- **tokens_per_second**: Generation throughput (higher is better)
- **input_tokens**: Tokens in your prompt
- **output_tokens**: Tokens in the response
- **total_tokens**: Sum of input + output (for cost calculation)

## 🔐 Security Notes

- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- In production, configure CORS with specific origins
- Consider adding rate limiting for production use
- Implement authentication/authorization as needed

## 📝 License

MIT License - Feel free to use this for your projects!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Support

For issues or questions, please open an issue on GitHub.

---

**Happy Coding! 🚀**
