#!/usr/bin/env python3
import requests


def test_ollama():
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "starcoder",  # or another model you have
        "prompt": "Write a simple hello world program in C.",
        "stream": False,
    }

    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        print("Success!")
        print(response.json()["response"])
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_ollama()
