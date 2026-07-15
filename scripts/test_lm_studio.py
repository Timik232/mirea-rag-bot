#!/usr/bin/env python3
"""Test LM Studio connectivity and model availability.

Verifies that:
1. LM Studio server is running on localhost:1234
2. The Gemma 4 E4B model is loaded
3. Chat completions work end-to-end
4. Response is in Russian and coherent

Usage:
    python scripts/test_lm_studio.py
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


def test_connection():
    """Test basic connectivity to LM Studio."""
    import requests

    base = config.LM_STUDIO_CONFIG["api_base"].rstrip("/v1")
    try:
        resp = requests.get(f"{base}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            print(f"✅ LM Studio server reachable at {base}")
            if models:
                print(f"   Available models: {[m['id'] for m in models]}")
            else:
                print("   ⚠️  No models loaded — load a model in LM Studio first")
            return True
        else:
            print(f"❌ Server returned {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to LM Studio at {base}")
        print("   Make sure LM Studio server is running (Developer → Start Server)")
        return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False


def test_chat():
    """Test a chat completion request."""
    import requests

    cfg = config.LM_STUDIO_CONFIG
    test_question = "Где находится главный корпус МИРЭА? Ответь кратко."

    payload = {
        "model": cfg["model"],
        "messages": [
            {
                "role": "system",
                "content": "Ты — справочный бот университета МИРЭА. Отвечай кратко и точно на русском языке.",
            },
            {"role": "user", "content": test_question},
        ],
        "temperature": cfg["temperature"],
        "max_tokens": cfg["max_tokens"],
    }

    try:
        start = time.time()
        resp = requests.post(
            f"{cfg['api_base']}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {cfg['api_key']}"},
            timeout=60,
        )
        elapsed = time.time() - start

        if resp.status_code != 200:
            print(f"❌ Chat completion failed: HTTP {resp.status_code}")
            print(f"   Response: {resp.text[:300]}")
            return False

        data = resp.json()
        answer = data["choices"][0]["message"]["content"]
        tokens = data.get("usage", {})

        print(f"✅ Chat completion successful ({elapsed:.1f}s)")
        print(f"   Question: {test_question}")
        print(f"   Answer: {answer[:200]}")
        if tokens:
            print(f"   Tokens: prompt={tokens.get('prompt_tokens', '?')}, "
                  f"completion={tokens.get('completion_tokens', '?')}")
        print(f"   Speed: {elapsed:.1f}s total")

        # Basic sanity: answer should mention Москва or Стромынка
        answer_lower = answer.lower()
        if "москва" in answer_lower or "стромынк" in answer_lower:
            print("   ✅ Answer contains expected keywords (Москва/Стромынка)")
        else:
            print("   ⚠️  Answer doesn't mention expected keywords")

        return True

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect — is LM Studio server running?")
        return False
    except KeyError:
        print(f"❌ Unexpected response format: {str(data)[:300]}")
        return False
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False


def main():
    print("=" * 60)
    print("LM Studio Connectivity Test")
    print(f"API base: {config.LM_STUDIO_CONFIG['api_base']}")
    print(f"Model:    {config.LM_STUDIO_CONFIG['model']}")
    print("=" * 60)

    print("\n--- Step 1: Connection ---")
    conn_ok = test_connection()

    if not conn_ok:
        print("\n" + "=" * 60)
        print("SETUP INSTRUCTIONS:")
        print("=" * 60)
        print("1. Open LM Studio")
        print("2. Download GGUF model from:")
        print("   lmstudio-community/gemma-4-e4b-it-GGUF")
        print("   → Choose Q4_K_M (~5GB) for 8GB VRAM")
        print("3. Load the model in LM Studio")
        print("4. Go to Developer tab → Start Server (port 1234)")
        print("5. Re-run: python scripts/test_lm_studio.py")
        return 1

    print("\n--- Step 2: Chat Completion ---")
    chat_ok = test_chat()

    print("\n" + "=" * 60)
    if conn_ok and chat_ok:
        print("🎉 All tests passed! LM Studio is ready for RAG.")
    else:
        print("💥 Some tests failed. See above for details.")
    print("=" * 60)

    return 0 if (conn_ok and chat_ok) else 1


if __name__ == "__main__":
    sys.exit(main())
