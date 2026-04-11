#!/bin/bash
# Example API calls for testing

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}LLM Chat API - Example Requests${NC}"
echo -e "${BLUE}========================================${NC}\n"

# 1. Health Check
echo -e "${GREEN}1. Health Check${NC}"
curl -s "$API_URL/" | json_pp
echo -e "\n"

# 2. Get Available Providers
echo -e "${GREEN}2. Get Available Providers${NC}"
curl -s "$API_URL/api/v1/providers" | json_pp
echo -e "\n"

# 3. Simple Chat - OpenAI
echo -e "${GREEN}3. Simple Chat - OpenAI${NC}"
curl -s -X POST "$API_URL/api/v1/chat/simple?message=Hello%20there&provider=openai" | json_pp
echo -e "\n"

# 4. Full Chat - OpenAI
echo -e "${GREEN}4. Full Chat Request - OpenAI${NC}"
curl -s -X POST "$API_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning in one sentence?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "provider": "openai"
  }' | json_pp
echo -e "\n"

# 5. Full Chat - Anthropic
echo -e "${GREEN}5. Full Chat Request - Anthropic${NC}"
curl -s -X POST "$API_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning in one sentence?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7,
    "provider": "anthropic"
  }' | json_pp
echo -e "\n"

# 6. Multi-turn Conversation
echo -e "${GREEN}6. Multi-turn Conversation${NC}"
curl -s -X POST "$API_URL/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is Python?"},
      {"role": "assistant", "content": "Python is a high-level programming language."},
      {"role": "user", "content": "What are its main features?"}
    ],
    "max_tokens": 200,
    "temperature": 0.7,
    "provider": "openai"
  }' | json_pp

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}Testing Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"
