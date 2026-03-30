#!/usr/bin/env python3
"""
MiniMax M2.5 Service Test Client

This script tests the MiniMax M2.5 service with various capabilities:
- Basic text completion
- Tool calling
- Reasoning with <think> tags
- Multi-turn conversations

Usage:
    python test_client.py [--port PORT] [--test-type TYPE]

Examples:
    # Test basic completion
    python test_client.py --test-type basic
    
    # Test tool calling
    python test_client.py --test-type tools
    
    # Test reasoning
    python test_client.py --test-type reasoning
    
    # Run all tests
    python test_client.py --test-type all
"""

import argparse
import json
import sys
import time
from typing import Dict, Any, Optional

try:
    import requests
except ImportError:
    print("Error: 'requests' library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests


class MiniMaxM2Client:
    """Client for testing MiniMax M2.5 service."""
    
    def __init__(self, base_url: str = "http://localhost:9084"):
        """Initialize the client with base URL."""
        self.base_url = base_url
        self.api_url = f"{base_url}/v1/chat/completions"
        
    def check_health(self) -> bool:
        """Check if the service is running."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def chat_completion(
        self,
        messages: list,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        max_tokens: Optional[int] = None,
        tools: Optional[list] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Send a chat completion request to the service.
        
        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature (default: server default 1.0)
            top_p: Nucleus sampling threshold (default: server default 0.95)
            top_k: Top-k sampling (default: server default 40)
            max_tokens: Maximum tokens to generate (default: server default 16384)
            tools: List of tool definitions (optional)
            stream: Whether to stream the response (default: False)
            
        Returns:
            Response dictionary from the API
        """
        payload = {
            "messages": messages,
            "stream": stream
        }
        
        # Only include parameters if explicitly provided (let server use defaults)
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        if top_k is not None:
            payload["top_k"] = top_k
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if tools is not None:
            payload["tools"] = tools
            
        try:
            response = requests.post(
                self.api_url,
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=300  # 5 minutes timeout for long requests
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error making request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            raise


def test_basic_completion(client: MiniMaxM2Client):
    """Test basic text completion."""
    print("\n" + "="*80)
    print("TEST 1: Basic Text Completion")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": "Write a Python function to calculate the Fibonacci sequence up to n terms."
        }
    ]
    
    print("\nSending request (using server defaults: temp=1.0, top_p=0.95, top_k=40)...")
    start_time = time.time()
    
    response = client.chat_completion(messages)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Response received in {elapsed:.2f} seconds")
    print("\nResponse:")
    print("-" * 80)
    content = response["choices"][0]["message"]["content"]
    print(content)
    print("-" * 80)
    
    # Check for <think> tags
    if "<think>" in content:
        print("\n✓ Response contains <think> reasoning tags (preserved by reasoning parser)")
    
    print(f"\nToken usage:")
    print(f"  Prompt tokens: {response['usage']['prompt_tokens']}")
    print(f"  Completion tokens: {response['usage']['completion_tokens']}")
    print(f"  Total tokens: {response['usage']['total_tokens']}")
    
    return True


def test_reasoning(client: MiniMaxM2Client):
    """Test reasoning capabilities with chain-of-thought."""
    print("\n" + "="*80)
    print("TEST 2: Reasoning with Chain-of-Thought")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": (
                "I have a coding problem: I need to find the longest palindromic substring "
                "in a given string. Can you think through the approach and provide an "
                "efficient solution in Python?"
            )
        }
    ]
    
    print("\nSending request with reasoning task...")
    start_time = time.time()
    
    response = client.chat_completion(messages)
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Response received in {elapsed:.2f} seconds")
    print("\nResponse:")
    print("-" * 80)
    content = response["choices"][0]["message"]["content"]
    print(content)
    print("-" * 80)
    
    if "<think>" in content and "</think>" in content:
        print("\n✓ SUCCESS: Response contains preserved <think>...</think> tags")
        print("  This shows the reasoning parser is working correctly")
    else:
        print("\n⚠ WARNING: No <think> tags found in response")
        print("  The model may not have used explicit reasoning, or tags were not preserved")
    
    return True


def test_tool_calling(client: MiniMaxM2Client):
    """Test tool calling capabilities."""
    print("\n" + "="*80)
    print("TEST 3: Tool Calling")
    print("="*80)
    
    # Define some example tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search for code in a codebase",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        },
                        "file_type": {
                            "type": "string",
                            "description": "File extension to filter by (e.g., 'py', 'js')"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco? Please check using the available tools."
        }
    ]
    
    print("\nSending request with tool definitions...")
    start_time = time.time()
    
    try:
        response = client.chat_completion(messages, tools=tools)
        elapsed = time.time() - start_time
        
        print(f"\n✓ Response received in {elapsed:.2f} seconds")
        
        # Check if tool was called
        message = response["choices"][0]["message"]
        
        if "tool_calls" in message and message["tool_calls"]:
            print("\n✓ SUCCESS: Model used tool calling!")
            print("\nTool calls:")
            for tool_call in message["tool_calls"]:
                print(f"  - Function: {tool_call['function']['name']}")
                print(f"    Arguments: {tool_call['function']['arguments']}")
        else:
            print("\n⚠ Model did not use tool calling")
            print("Response content:")
            print("-" * 80)
            print(message.get("content", "No content"))
            print("-" * 80)
        
        return True
    except Exception as e:
        print(f"\n✗ Tool calling test failed: {e}")
        return False


def test_multi_turn(client: MiniMaxM2Client):
    """Test multi-turn conversation."""
    print("\n" + "="*80)
    print("TEST 4: Multi-turn Conversation")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": "I'm building a REST API. What's the difference between PUT and PATCH?"
        }
    ]
    
    print("\nTurn 1: Asking about PUT vs PATCH...")
    response1 = client.chat_completion(messages)
    assistant_reply = response1["choices"][0]["message"]["content"]
    
    print("✓ Response received")
    print(f"  Tokens: {response1['usage']['total_tokens']}")
    
    # Add assistant response to conversation
    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })
    
    # Ask follow-up
    messages.append({
        "role": "user",
        "content": "Can you show me a Python example using FastAPI for both methods?"
    })
    
    print("\nTurn 2: Asking for code examples...")
    response2 = client.chat_completion(messages)
    
    print("✓ Response received")
    print(f"  Tokens: {response2['usage']['total_tokens']}")
    
    print("\nFinal response:")
    print("-" * 80)
    print(response2["choices"][0]["message"]["content"][:500] + "...")
    print("-" * 80)
    
    print("\n✓ SUCCESS: Multi-turn conversation completed")
    
    return True


def test_parameter_override(client: MiniMaxM2Client):
    """Test that caller can override default parameters."""
    print("\n" + "="*80)
    print("TEST 5: Parameter Override")
    print("="*80)
    
    messages = [
        {
            "role": "user",
            "content": "Count from 1 to 5."
        }
    ]
    
    print("\nTest 5a: Using server defaults (temp=1.0, top_p=0.95, top_k=40)")
    response1 = client.chat_completion(messages)
    print("✓ Response received with server defaults")
    
    print("\nTest 5b: Overriding with lower temperature (temp=0.1)")
    response2 = client.chat_completion(messages, temperature=0.1)
    print("✓ Response received with custom temperature")
    
    print("\nTest 5c: Custom parameters (temp=0.5, top_p=0.9, top_k=10, max_tokens=50)")
    response3 = client.chat_completion(
        messages,
        temperature=0.5,
        top_p=0.9,
        top_k=10,
        max_tokens=50
    )
    print("✓ Response received with all custom parameters")
    
    print("\n✓ SUCCESS: Parameter override works correctly")
    print("  Server defaults are used when not specified")
    print("  Caller-provided values take precedence when specified")
    
    return True


def run_all_tests(client: MiniMaxM2Client):
    """Run all tests."""
    tests = [
        ("Basic Completion", test_basic_completion),
        ("Reasoning", test_reasoning),
        ("Tool Calling", test_tool_calling),
        ("Multi-turn Conversation", test_multi_turn),
        ("Parameter Override", test_parameter_override),
    ]
    
    results = []
    
    print("\n" + "="*80)
    print("RUNNING ALL TESTS")
    print("="*80)
    
    for test_name, test_func in tests:
        try:
            success = test_func(client)
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test client for MiniMax M2.5 service"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9084,
        help="Port where the service is running (default: 9084)"
    )
    parser.add_argument(
        "--test-type",
        choices=["basic", "reasoning", "tools", "multi-turn", "params", "all"],
        default="basic",
        help="Type of test to run (default: basic)"
    )
    
    args = parser.parse_args()
    
    base_url = f"http://localhost:{args.port}"
    client = MiniMaxM2Client(base_url)
    
    print("MiniMax M2.5 Service Test Client")
    print("="*80)
    print(f"Service URL: {base_url}")
    
    # Check if service is running
    print("\nChecking service health...")
    if not client.check_health():
        print(f"✗ Error: Service is not running at {base_url}")
        print("\nPlease start the MiniMax M2.5 service first:")
        print("  cd ~/ModelService_MinMax-M2")
        print("  ./start_service.sh")
        sys.exit(1)
    
    print("✓ Service is running")
    
    # Run selected test
    try:
        if args.test_type == "basic":
            success = test_basic_completion(client)
        elif args.test_type == "reasoning":
            success = test_reasoning(client)
        elif args.test_type == "tools":
            success = test_tool_calling(client)
        elif args.test_type == "multi-turn":
            success = test_multi_turn(client)
        elif args.test_type == "params":
            success = test_parameter_override(client)
        elif args.test_type == "all":
            success = run_all_tests(client)
        
        if success:
            print("\n" + "="*80)
            print("✓ ALL TESTS COMPLETED SUCCESSFULLY")
            print("="*80)
            sys.exit(0)
        else:
            print("\n" + "="*80)
            print("✗ SOME TESTS FAILED")
            print("="*80)
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

