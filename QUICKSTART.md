# Quick Start Guide

## Setup in 5 Minutes

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys
```bash
cp .env.example .env
```

Then edit `.env` and add at least one API key:
```
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Step 3: Run the Server
```bash
uvicorn app.main:app --reload
```

### Step 4: Test It
Open your browser to: http://localhost:8000/docs

Or use curl:
```bash
curl -X POST "http://localhost:8000/api/v1/chat/simple?message=Hello"
```

## Testing Performance

Run the included performance test:
```bash
python test_performance.py
```

## Interactive Testing

Visit the interactive API docs:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Example Requests

### Simple Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat/simple?message=What%20is%20AI&provider=openai"
```

### Full Chat
```bash
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "provider": "openai",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

## Troubleshooting

### "API key not configured"
- Make sure you created `.env` file
- Check that your API key is correct
- Verify the key starts with `sk-` for OpenAI or `sk-ant-` for Anthropic

### "Module not found"
- Make sure you're in the virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

### Port already in use
- Change the port: `uvicorn app.main:app --reload --port 8001`

## Next Steps

1. ✅ Test both providers (OpenAI and Anthropic)
2. ✅ Run performance comparison tests
3. ✅ Try different prompts and parameters
4. ✅ Deploy to Render (see README.md)
5. ✅ Build your frontend application!

## Cost Monitoring

Each response includes token usage:
```json
{
  "usage": {
    "input_tokens": 15,
    "output_tokens": 106,
    "total_tokens": 121
  }
}
```

Use this to track costs:
- OpenAI GPT-4 Turbo: ~$0.01 per 1K input tokens, ~$0.03 per 1K output tokens
- Anthropic Claude: ~$0.003 per 1K input tokens, ~$0.015 per 1K output tokens

## Need Help?

Check the main README.md for detailed documentation!
