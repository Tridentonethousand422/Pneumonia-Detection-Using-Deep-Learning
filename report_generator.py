"""
=============================================================================
  AI-Powered Preliminary Radiology Report Generator
  Local LLM Integration via Ollama (Qwen 2.5)
=============================================================================

  This module sends the CNN's classification output to a locally-hosted
  Qwen 2.5 LLM running on Ollama. The LLM drafts a structured,
  professional preliminary radiology report.

  IMPORTANT — RESOURCE ISOLATION STRATEGY
  ─────────────────────────────────────────
  Ollama runs as a separate process and manages its own memory.
  By default, Ollama loads models onto the GPU. Since the TensorFlow
  vision model already occupies VRAM, set the following environment
  variable BEFORE starting Ollama to force CPU-only inference:

      export CUDA_VISIBLE_DEVICES=""   # then: ollama serve

  This ensures the LLM and CNN never compete for VRAM on the RTX 5050.
=============================================================================
"""

import requests
import json
import os
from typing import Optional

# ── Configuration ────────────────────────────────────────────────────────────
DEFAULT_MODEL  = "qwen2.5"


def _detect_ollama_url() -> str:
    """
    Auto-detect the Ollama API URL, handling WSL2 → Windows networking.

    When Ollama runs on Windows (PowerShell) but Python runs inside WSL2,
    'localhost' in WSL2 refers to the Linux VM — not the Windows host.
    This function reads the Windows host IP from the default gateway
    (via `ip route`), which reliably points to the Windows host.

    Priority:
      1. OLLAMA_HOST environment variable (if set by the user)
      2. Windows host IP from default gateway (WSL2 auto-detection)
      3. Fallback to localhost (native Linux or mirrored networking)
    """
    # 1. User-defined override via environment variable
    env_host = os.environ.get("OLLAMA_HOST")
    if env_host:
        base = env_host if env_host.startswith("http") else f"http://{env_host}"
        return f"{base.rstrip('/')}/api/generate"

    # 2. WSL2 auto-detection: read the Windows host IP from the default gateway
    #    The default gateway in WSL2 always points to the Windows host.
    #    (Note: /etc/resolv.conf is unreliable — newer WSL2 uses a DNS relay IP)
    try:
        import subprocess
        result = subprocess.run(
            ["ip", "route", "show", "default"],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0 and "via" in result.stdout:
            host_ip = result.stdout.strip().split()[2]
            return f"http://{host_ip}:11434/api/generate"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # Not running in WSL2

    # 3. Fallback (native Linux, Docker, or WSL2 with mirrored networking)
    return "http://localhost:11434/api/generate"


OLLAMA_API_URL = _detect_ollama_url()


def generate_report(
    prediction: str,
    confidence: float,
    model_name: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_API_URL,
    timeout: int = 90,
) -> str:
    """
    Generate an AI preliminary radiology report via a local LLM.

    The function constructs a structured prompt containing the CNN's
    classification result and confidence score, sends it to the Ollama
    REST API, and returns the LLM's generated text.

    Args:
        prediction : "PNEUMONIA" or "NORMAL"
        confidence : Float in [0, 1] representing model certainty
        model_name : Ollama model identifier (default: "qwen2.5")
        ollama_url : Ollama API endpoint URL
        timeout    : HTTP request timeout in seconds

    Returns:
        report_text : The generated report string, or an error message
                      if the LLM service is unavailable.
    """
    confidence_pct = confidence * 100

    # ── Structured prompt for medical report generation ──────────────────────
    prompt = (
        "You are a radiologist AI assistant. Based on the following "
        "deep learning analysis of a posterior-anterior (PA) chest X-ray, "
        "draft a concise, professional preliminary radiology report.\n\n"
        "═══════════════════════════════════════\n"
        "  CNN ANALYSIS RESULTS\n"
        "═══════════════════════════════════════\n"
        f"  Classification : {prediction}\n"
        f"  Confidence     : {confidence_pct:.1f}%\n"
        f"  Model          : MobileNetV2 (Fine-Tuned)\n"
        "═══════════════════════════════════════\n\n"
        "Report format instructions:\n"
        "1. Title: 'AI-GENERATED PRELIMINARY REPORT'\n"
        "2. Section 'EXAMINATION': State 'PA Chest Radiograph'\n"
        "3. Section 'FINDINGS': Describe what the classification implies "
        "about the lung fields. If PNEUMONIA, mention possible opacities; "
        "if NORMAL, mention clear lung fields.\n"
        "4. Section 'IMPRESSION': One-line clinical summary.\n"
        "5. Section 'RECOMMENDATIONS': Standard clinical follow-up steps.\n"
        "6. Footer disclaimer: Clearly state this is AI-generated and "
        "requires review by a board-certified radiologist.\n\n"
        "Keep the report under 200 words. Be professional and concise."
    )

    # ── Send request to Ollama ───────────────────────────────────────────────
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,         # Low temperature for factual output
            "num_predict": 512,          # Max tokens to generate
        },
    }

    try:
        response = requests.post(ollama_url, json=payload, timeout=timeout)
        response.raise_for_status()

        result = response.json()
        report = result.get("response", "").strip()

        if not report:
            return "⚠️ The LLM returned an empty response. Please try again."

        return report

    except requests.ConnectionError:
        return (
            "⚠️ **Could not connect to Ollama.**\n\n"
            "Please ensure Ollama is running:\n"
            "```bash\n"
            "# Start Ollama on CPU to preserve GPU VRAM for TensorFlow\n"
            "CUDA_VISIBLE_DEVICES=\"\" ollama serve\n"
            "```\n\n"
            f"Then pull the model: `ollama pull {model_name}`"
        )

    except requests.Timeout:
        return (
            "⚠️ **Request timed out.**\n\n"
            "The LLM may be loading into memory for the first time. "
            "Please wait a moment and try again."
        )

    except requests.HTTPError as e:
        return f"⚠️ **Ollama HTTP error:** {e}"

    except Exception as e:
        return f"⚠️ **Report generation failed:** {str(e)}"


# ── Quick self-test ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing report generation with PNEUMONIA @ 94.2% confidence...\n")
    print(generate_report("PNEUMONIA", 0.942))
