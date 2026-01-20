#!/usr/bin/env python3
"""
Quick Test Script for Production RAG System
Tests basic functionality of all components
"""

import requests
import time
import json
from typing import Dict, Any

# Service endpoints
SERVICES = {
    "RAG App": "http://localhost:8000",
    "Airflow": "http://localhost:8080/health",
    "Redis": "http://localhost:6379",
    "RedisInsight": "http://localhost:8001",
    "ChromaDB": "http://localhost:8003/api/v1/heartbeat",
    "vLLM Vision": "http://localhost:8006/v1/models",
    "vLLM Reasoning": "http://localhost:8005/v1/models",
}

def check_service(name: str, url: str, timeout: int = 5) -> bool:
    """Check if a service is responding."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code < 400:
            print(f"✅ {name:20} - OK ({response.status_code})")
            return True
        else:
            print(f"❌ {name:20} - HTTP {response.status_code}")
            return False
    except requests.exceptions.Timeout:
        print(f"⏱️  {name:20} - Timeout")
        return False
    except requests.exceptions.ConnectionError:
        print(f"❌ {name:20} - Connection refused")
        return False
    except Exception as e:
        print(f"❌ {name:20} - {str(e)}")
        return False

def test_chromadb_collection():
    """Test ChromaDB collection access."""
    try:
        response = requests.get("http://localhost:8003/api/v1/collections")
        if response.status_code == 200:
            collections = response.json()
            print(f"\n📊 ChromaDB Collections:")
            if collections:
                for col in collections:
                    print(f"   • {col.get('name', 'unknown')}")
            else:
                print(f"   (empty - no collections yet)")
            return True
        return False
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_redis_connection():
    """Test Redis connection."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, decode_responses=True)
        r.ping()
        info = r.info()
        print(f"\n📊 Redis Stats:")
        print(f"   • Memory Used: {info.get('used_memory_human', 'unknown')}")
        print(f"   • Connected Clients: {info.get('connected_clients', 0)}")
        print(f"   • Uptime: {info.get('uptime_in_seconds', 0)} seconds")
        return True
    except Exception as e:
        print(f"❌ Redis test failed: {e}")
        return False

def test_vllm_inference():
    """Test vLLM model inference."""
    try:
        # Test reasoning model
        response = requests.post(
            "http://localhost:8005/v1/chat/completions",
            json={
                "model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "messages": [{"role": "user", "content": "Say 'test'"}],
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            print(f"\n🤖 vLLM Inference Test:")
            print(f"   • Response: {content[:50]}...")
            return True
        return False
    except Exception as e:
        print(f"❌ vLLM test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("Production RAG System - Health Check")
    print("="*60)
    print()
    
    results = {}
    
    # Test all services
    print("🔍 Checking Services...")
    print("-"*60)
    for name, url in SERVICES.items():
        results[name] = check_service(name, url)
    
    print()
    print("-"*60)
    
    # Additional tests
    if results.get("ChromaDB", False):
        test_chromadb_collection()
    
    if results.get("Redis", False):
        test_redis_connection()
    
    if results.get("vLLM Reasoning", False):
        test_vllm_inference()
    
    # Summary
    print()
    print("="*60)
    total = len(results)
    passed = sum(results.values())
    print(f"Summary: {passed}/{total} services healthy")
    
    if passed == total:
        print("✅ All systems operational!")
        return 0
    elif passed >= total * 0.7:
        print("⚠️  Some services down, but core functionality available")
        return 1
    else:
        print("❌ Multiple services down - check logs")
        return 2

if __name__ == "__main__":
    exit(main())
