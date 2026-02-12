#!/usr/bin/env python3
"""
Test an LLM on medical consultation scenarios.

This script runs an LLM through the CaseSim API to evaluate its
consultation capabilities against malpractice case ground truth.

Usage:
    # Test with OpenAI
    python scripts/test_llm.py --provider openai --model gpt-4o

    # Test with Claude
    python scripts/test_llm.py --provider anthropic --model claude-3-opus-20240229

    # Test specific case
    python scripts/test_llm.py --provider openai --model gpt-4o --case-id bailii-qb-2021-169-html

    # Test multiple cases
    python scripts/test_llm.py --provider openai --model gpt-4o --num-cases 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import httpx

# Add gui directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gui"))

# Log directory
LOG_DIR = Path(__file__).parent.parent / "gui" / "logs"


def get_completed_cases(llm_name: str) -> set[str]:
    """Get set of case_ids already tested by this LLM."""
    completed = set()
    if not LOG_DIR.exists():
        return completed

    for log_file in LOG_DIR.glob("*.json"):
        try:
            with open(log_file, "r") as f:
                data = json.load(f)
            if data.get("llm_name") == llm_name:
                completed.add(data.get("case_id"))
        except (json.JSONDecodeError, OSError):
            continue

    return completed


# LLM system prompt for medical consultation
CONSULTATION_SYSTEM_PROMPT = """You are a doctor conducting a medical consultation with a patient.

CRITICAL RULE - ONE QUESTION PER MESSAGE:
You MUST ask only ONE question per message. The patient can only answer one question at a time.
After the patient answers, you can ask your next question.
NEVER number questions or ask multiple questions. Just ask ONE.

Your process:
1. Gather history by asking ONE question, waiting for answer, then asking next question
2. Order tests when appropriate (say "I'd like to order [test name]")
3. When ready, provide your assessment and recommendations

Example of CORRECT behavior:
- "Can you describe the nature of your chest pain?"
- (wait for answer)
- "How long have you had this pain?"
- (wait for answer)

Example of WRONG behavior (DO NOT DO THIS):
- "Can you describe the pain? How long have you had it? Do you have any other symptoms?"

Start now by asking the patient ONE question about their symptoms."""


def get_llm_client(provider: str, model: str):
    """Get the appropriate LLM client."""
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    elif provider == "anthropic":
        from anthropic import Anthropic
        return Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    elif provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        return client
    elif provider == "together":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("TOGETHER_API_KEY"),
            base_url="https://api.together.xyz/v1"
        )
    elif provider == "groq":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )
    elif provider == "huggingface":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("HF_API_KEY"),
            base_url="https://router.huggingface.co/v1"
        )
    elif provider == "openrouter":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
    elif provider == "dr7":
        from openai import OpenAI
        return OpenAI(
            api_key=os.environ.get("DR7_API_KEY"),
            base_url="https://api.dr7.ai/v1"
        )
    elif provider == "vllm":
        from openai import OpenAI
        # Custom vLLM endpoint - set VLLM_BASE_URL env var
        base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        return OpenAI(
            api_key="EMPTY",  # vLLM doesn't require auth by default
            base_url=base_url
        )
    elif provider == "vertexai":
        from google.auth import default
        import google.auth.transport.requests
        from openai import OpenAI

        # Get project and location from env
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not project_id:
            raise ValueError("GOOGLE_CLOUD_PROJECT environment variable required for vertexai provider")

        # Get access token via Application Default Credentials
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        credentials.refresh(google.auth.transport.requests.Request())

        return OpenAI(
            base_url=f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/endpoints/openapi",
            api_key=credentials.token,
        )
    elif provider == "replicate":
        import replicate
        # Replicate client is module-level, just return the module
        return replicate
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_llm_response(client, provider: str, model: str, messages: list[dict]) -> str:
    """Get response from LLM."""
    if provider in ["openai", "together", "groq", "huggingface", "openrouter", "dr7", "vertexai", "vllm"]:
        # Newer models (gpt-4o, gpt-5, etc.) use max_completion_tokens
        # Older models use max_tokens
        # Gemini 3 and DeepSeek R1 with reasoning need more tokens
        max_tokens = 16000 if ("gemini-3" in model or "deepseek-r1" in model) else 1500
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_completion_tokens=max_tokens,
            )
        except Exception as e:
            if "max_completion_tokens" in str(e):
                # Fallback for older models
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1500,
                )
            else:
                raise
        message = response.choices[0].message
        content = message.content

        # Strip <think>...</think> reasoning tags from DeepSeek R1 and Qwen3 responses
        if content and ("deepseek-r1" in model or "qwen3" in model):
            import re
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

        # Debug: print full message structure for newer models
        if content is None or content == "":
            print(f"\n[DEBUG] Empty response detected")
            print(f"[DEBUG] Full message: {message}")
            print(f"[DEBUG] Finish reason: {response.choices[0].finish_reason}")
            refusal = getattr(message, 'refusal', None)
            if refusal:
                return f"[Model refused: {refusal}]"
            return "[No response from model]"
        return content
    elif provider == "anthropic":
        # Convert messages to Anthropic format
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        conversation = [m for m in messages if m["role"] != "system"]

        response = client.messages.create(
            model=model,
            system=system,
            messages=conversation,
            max_tokens=1500,
        )
        return response.content[0].text
    elif provider == "google":
        import time
        from google.genai import types
        from google.genai.errors import ClientError

        # Convert messages to Gemini format
        system_instruction = next((m["content"] for m in messages if m["role"] == "system"), None)

        # Build contents list for Gemini
        contents = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] in ["user", "patient"] else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=m["content"])]
            ))

        # Retry with backoff for rate limits
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.3,
                        max_output_tokens=1500,
                    ),
                )
                return response.text
            except ClientError as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    wait_time = 2 ** attempt * 10  # 10, 20, 40, 80, 160 seconds
                    print(f"\n[Rate limited, waiting {wait_time}s before retry {attempt + 1}/{max_retries}]")
                    time.sleep(wait_time)
                else:
                    raise
        raise Exception("Max retries exceeded for Gemini API")
    elif provider == "replicate":
        # Replicate uses a different API format
        # Convert chat messages to a single prompt
        system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")

        # Build conversation history
        conversation = []
        for m in messages:
            if m["role"] == "system":
                continue
            elif m["role"] == "user":
                conversation.append(f"Patient: {m['content']}")
            elif m["role"] == "assistant":
                conversation.append(f"Doctor: {m['content']}")

        prompt = "\n".join(conversation)
        if conversation:
            prompt += "\nDoctor:"

        # Map model names to Replicate model IDs
        model_map = {
            "meditron-70b": "titocosta/meditron-70b-awq:7cbcd02ebd1baa7f800969f60dada8bd33c30e8d467223ce78842ecba2fbbc86",
            "meditron-7b": "titocosta/meditron:621950ce014dc57c977d3a92e8aa17fa4aba31a4c9bf21f06c0655f4e6f73e58",
        }

        replicate_model = model_map.get(model, model)

        output = client.run(
            replicate_model,
            input={
                "prompt": prompt,
                "system_message": system_msg,
                "max_new_tokens": 1500,
                "temperature": 0.3,
                "top_p": 0.95,
            }
        )

        # Output is a generator of strings, concatenate them
        result = "".join(output)
        return result.strip()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def run_consultation(
    api_base: str,
    client,
    provider: str,
    model: str,
    case_id: str | None = None,
    max_turns: int = 10,
) -> dict:
    """Run a consultation session with an LLM."""

    # Start session
    start_data = {"llm_name": f"{provider}/{model}"}
    if case_id:
        start_data["case_id"] = case_id

    response = httpx.post(f"{api_base}/sessions", json=start_data, timeout=30.0)
    response.raise_for_status()
    session = response.json()

    session_id = session["session_id"]
    presentation = session["presentation"]

    print(f"\n{'='*60}")
    print(f"Case: {session['case_id']}")
    print(f"{'='*60}")
    print(f"\nPresentation:\n{presentation}\n")

    # Initialize conversation for LLM
    messages = [
        {"role": "system", "content": CONSULTATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Patient presentation:\n\n{presentation}"},
    ]

    # Run consultation loop
    turn = 0
    final_recommendation = None

    while turn < max_turns:
        turn += 1

        # Get LLM response
        llm_response = get_llm_response(client, provider, model, messages)
        print(f"\n[Doctor (Turn {turn})]:\n{llm_response}")

        # Check if LLM is providing final recommendation
        final_indicators = [
            "my assessment is",
            "my diagnosis is",
            "i recommend",
            "my recommendation",
            "final diagnosis",
            "in conclusion",
            "to summarize",
            "based on my assessment",
        ]

        # Handle None responses from LLM
        if not llm_response:
            print("Warning: LLM returned empty response, asking for clarification")
            llm_response = "I need more information. Can you describe your symptoms in more detail?"

        is_final = any(ind in llm_response.lower() for ind in final_indicators)

        if is_final and turn >= 3:  # At least 3 turns before final
            final_recommendation = llm_response
            break

        # Send to API
        chat_response = httpx.post(
            f"{api_base}/sessions/{session_id}/chat",
            json={"message": llm_response},
            timeout=60.0,
        )
        chat_response.raise_for_status()
        patient_response = chat_response.json()

        print(f"\n[Patient]:\n{patient_response['response']}")

        # Add to conversation
        messages.append({"role": "assistant", "content": llm_response})
        messages.append({"role": "user", "content": patient_response["response"]})

    # End session
    if not final_recommendation:
        # Ask LLM for final recommendation
        messages.append({
            "role": "user",
            "content": "Based on our conversation, please provide your final assessment and recommendations."
        })
        final_recommendation = get_llm_response(client, provider, model, messages)
        print(f"\n[Doctor - Final]:\n{final_recommendation}")

    # Handle invalid recommendations (some models return "None" or empty strings)
    if not final_recommendation or final_recommendation.lower().strip() in ["none", "n/a", ""]:
        final_recommendation = "Unable to provide specific recommendation based on available information. Further evaluation needed."

    end_response = httpx.post(
        f"{api_base}/sessions/{session_id}/end",
        json={"final_recommendation": final_recommendation},
        timeout=120.0,  # Evaluation can take a while due to LLM calls
    )
    end_response.raise_for_status()
    result = end_response.json()

    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")
    print(f"Score: {result['score']}/2")
    print(f"Legally Defensible: {result['legally_defensible']}")
    print(f"\nFeedback:\n{result['feedback']}")
    print(f"\nLog saved to: {result['conversation_log_path']}")

    # Include case_id in result for tracking
    result["case_id"] = session["case_id"]
    return result


def main():
    parser = argparse.ArgumentParser(description="Test an LLM on medical consultations")
    parser.add_argument("--provider", choices=["openai", "anthropic", "google", "together", "groq", "huggingface", "openrouter", "dr7", "vertexai", "replicate", "vllm"], default="openai")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--case-id", help="Specific case ID to test")
    parser.add_argument("--num-cases", type=int, default=1, help="Number of random cases to test")
    parser.add_argument("--max-turns", type=int, default=10, help="Max turns per consultation")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers")
    parser.add_argument("--output", help="Output file for results")

    args = parser.parse_args()

    # Get LLM client
    llm_name = f"{args.provider}/{args.model}"
    client = get_llm_client(args.provider, args.model)

    # Get already completed cases for this LLM
    completed_cases = get_completed_cases(llm_name)
    if completed_cases:
        print(f"Found {len(completed_cases)} cases already tested with {llm_name}")

    # Get available cases from API
    cases_response = httpx.get(f"{args.api_base}/cases", timeout=30.0)
    cases_response.raise_for_status()
    available_cases = cases_response.json()["cases"]
    available_case_ids = [c["case_id"] for c in available_cases]

    # Filter out completed cases
    untested_case_ids = [cid for cid in available_case_ids if cid not in completed_cases]
    print(f"Available cases: {len(available_case_ids)}, Untested: {len(untested_case_ids)}")

    results = []

    if args.case_id:
        # Test specific case (even if already done)
        if args.case_id in completed_cases:
            print(f"Warning: Case {args.case_id} was already tested with {llm_name}")
        result = run_consultation(
            args.api_base, client, args.provider, args.model,
            case_id=args.case_id, max_turns=args.max_turns
        )
        results.append(result)
    else:
        # Test multiple cases, skipping already completed ones
        if not untested_case_ids:
            print("All available cases have already been tested with this LLM.")
            return

        num_to_test = min(args.num_cases, len(untested_case_ids))
        if num_to_test < args.num_cases:
            print(f"Only {num_to_test} untested cases available (requested {args.num_cases})")

        cases_to_test = untested_case_ids[:num_to_test]

        if args.workers > 1:
            # Parallel execution
            print_lock = threading.Lock()
            completed_count = [0]

            def run_case(case_id):
                # Each worker gets its own client to avoid thread-safety issues
                worker_client = get_llm_client(args.provider, args.model)
                try:
                    result = run_consultation(
                        args.api_base, worker_client, args.provider, args.model,
                        case_id=case_id, max_turns=args.max_turns
                    )
                    with print_lock:
                        completed_count[0] += 1
                        print(f"\n[Progress: {completed_count[0]}/{num_to_test} completed]")
                    return result
                except Exception as e:
                    with print_lock:
                        completed_count[0] += 1
                        print(f"\n[FAILED {case_id}: {e}] ({completed_count[0]}/{num_to_test})")
                    return None

            print(f"\nRunning {num_to_test} cases with {args.workers} parallel workers")
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(run_case, cid): cid for cid in cases_to_test}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
        else:
            # Sequential execution (original behavior)
            for i, case_id in enumerate(cases_to_test):
                print(f"\n{'#'*60}")
                print(f"Test {i+1}/{num_to_test}")
                print(f"{'#'*60}")

                result = run_consultation(
                    args.api_base, client, args.provider, args.model,
                    case_id=case_id, max_turns=args.max_turns
                )
                results.append(result)

    if not results:
        return

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"LLM: {llm_name}")
    print(f"Cases tested: {len(results)}")

    scores = [r["score"] for r in results]
    print(f"Average score: {sum(scores)/len(scores):.2f}/2")
    print(f"Legally defensible: {sum(1 for r in results if r['legally_defensible'])}/{len(results)}")

    # Save results
    if args.output:
        output_data = {
            "provider": args.provider,
            "model": args.model,
            "tested_at": datetime.utcnow().isoformat(),
            "results": results,
            "summary": {
                "cases_tested": len(results),
                "average_score": sum(scores)/len(scores),
                "legally_defensible_count": sum(1 for r in results if r["legally_defensible"]),
            }
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
