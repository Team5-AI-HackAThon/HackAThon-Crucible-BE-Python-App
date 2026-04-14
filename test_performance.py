import httpx
import asyncio
import json
import statistics
from typing import List

async def test_latency(
    url: str,
    message: str,
    provider: str,
    num_requests: int = 5
):
    """Test API latency with multiple requests"""
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        results = []
        
        print(f"\n{'='*60}")
        print(f"Testing {provider.upper()} with {num_requests} requests")
        print(f"{'='*60}\n")
        
        for i in range(num_requests):
            payload = {
                "messages": [{"role": "user", "content": message}],
                "max_tokens": 500,
                "temperature": 0.7,
                "provider": provider
            }
            
            try:
                response = await client.post(
                    f"{url}/api/v1/chat",
                    json=payload
                )
                
                if response.status_code == 200:
                    data = response.json()
                    perf = data["performance"]
                    
                    results.append({
                        "total_time": perf["total_time_seconds"],
                        "tokens_per_second": perf["tokens_per_second"],
                        "output_tokens": perf["output_tokens"]
                    })
                    
                    print(f"Request {i+1}:")
                    print(f"  ⏱️  Total Time: {perf['total_time_seconds']:.4f}s")
                    print(f"  🚀 Tokens/sec: {perf['tokens_per_second']:.2f}")
                    print(f"  📝 Output Tokens: {perf['output_tokens']}")
                    print(f"  📊 Total Tokens: {perf['total_tokens']}\n")
                else:
                    print(f"Request {i+1} failed: {response.status_code}")
                    print(f"Response: {response.text}\n")
                    
            except Exception as e:
                print(f"Request {i+1} error: {str(e)}\n")
        
        # Calculate statistics
        if results:
            times = [r["total_time"] for r in results]
            tps = [r["tokens_per_second"] for r in results]
            
            print(f"\n{'='*60}")
            print(f"STATISTICS for {provider.upper()}")
            print(f"{'='*60}")
            print(f"Average Response Time: {statistics.mean(times):.4f}s")
            print(f"Min Response Time: {min(times):.4f}s")
            print(f"Max Response Time: {max(times):.4f}s")
            if len(times) > 1:
                print(f"Std Dev: {statistics.stdev(times):.4f}s")
            print(f"\nAverage Tokens/sec: {statistics.mean(tps):.2f}")
            print(f"Min Tokens/sec: {min(tps):.2f}")
            print(f"Max Tokens/sec: {max(tps):.2f}")
            print(f"{'='*60}\n")
        else:
            print(f"❌ No successful requests for {provider.upper()}\n")

async def compare_providers(url: str, message: str):
    """Compare performance between providers"""
    
    # Test which providers are available
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{url}/api/v1/providers")
            if response.status_code == 200:
                data = response.json()
                providers = data.get("available_providers", [])
                print(f"\n🔍 Available Providers: {', '.join(providers)}")
                
                # Test each available provider
                for provider in providers:
                    await test_latency(url, message, provider, num_requests=5)
            else:
                print(f"❌ Failed to get providers: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking providers: {str(e)}")

if __name__ == "__main__":
    # Configuration
    API_URL = "http://localhost:8000"  # Change to your deployed URL for production testing
    TEST_MESSAGE = "Explain quantum computing in simple terms."
    
    print("\n" + "="*60)
    print("LLM Chat API - Performance Testing")
    print("="*60)
    print(f"API URL: {API_URL}")
    print(f"Test Message: {TEST_MESSAGE}")
    
    # Run tests
    asyncio.run(compare_providers(API_URL, TEST_MESSAGE))
