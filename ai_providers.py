import os
import logging
import json
import requests

logger = logging.getLogger(__name__)


def _try_gemini(prompt):
    try:
        import google.genai as genai
    except Exception:
        return None

    try:
        # Prefer GenerativeModel if available
        if hasattr(genai, 'GenerativeModel'):
            model = genai.GenerativeModel('gemini-2.5-flash')
            resp = model.generate_content(prompt)
            return getattr(resp, 'text', None) or getattr(resp, 'response', None) or str(resp)

        # Fallback to a Client API if present
        if hasattr(genai, 'Client'):
            try:
                client = genai.Client()
                if hasattr(client, 'generate'):
                    resp = client.generate(prompt)
                    return getattr(resp, 'text', None) or str(resp)
                if hasattr(client, 'generate_text'):
                    resp = client.generate_text(prompt)
                    return getattr(resp, 'text', None) or str(resp)
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"Gemini provider failed: {e}")
    return None


def _try_groq(prompt):
    # Attempt to call Groq via SDK or HTTP. Requires GROQ_API_KEY and GROQ_MODEL
    groq_key = os.environ.get('GROQ_API_KEY') or os.environ.get('GROQ_KEY')
    groq_model = os.environ.get('GROQ_MODEL')
    if not groq_key or not groq_model:
        return None
    # Try SDK if installed (preferred, using Chat Completions API)
    try:
        from groq import Groq
        try:
            client = Groq(api_key=groq_key)
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            resp = client.chat.completions.create(messages=messages, model=groq_model)
            # Response shape per docs: resp.choices[0].message.content
            try:
                return resp.choices[0].message.content
            except Exception:
                return str(resp)
        except Exception as e:
            logger.debug(f"Groq SDK chat call failed: {e}")
    except Exception:
        logger.debug("Groq SDK not installed")

    # Fallback to HTTP request (best-effort). The exact Groq API path may differ.
    try:
        base = os.environ.get('GROQ_API_URL', 'https://api.groq.ai/v1')
        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        payload = {"model": groq_model, "messages": [{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":prompt}], "max_completion_tokens": 1024}
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code == 200:
            data = r.json()
            # Try common fields
            if isinstance(data, dict):
                # docs show choices[0].message.content
                try:
                    return data['choices'][0]['message']['content']
                except Exception:
                    return json.dumps(data)
            return str(data)
    except Exception as e:
        logger.debug(f"Groq provider HTTP call failed: {e}")

    return None


def generate_text(prompt, prefer=None):
    """Generate text from available providers.

    prefer: list or tuple ordering provider names e.g. ('gemini','groq')
    Returns generated text or raises RuntimeError if none available.
    """
    if prefer is None:
        prefer = ('gemini', 'groq')

    for p in prefer:
        if p == 'gemini':
            out = _try_gemini(prompt)
            if out:
                return out
        elif p == 'groq':
            out = _try_groq(prompt)
            if out:
                return out

    raise RuntimeError('No LLM providers available or all providers failed')
