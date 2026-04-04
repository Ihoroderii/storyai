import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def call_llm(prompt: str, provider: str = None, model: str = None) -> str:
    """
    Calls LLM API to generate text. Supports multiple providers.
    
    Provider priority (uses first available):
    1. ANTHROPIC_API_KEY -> Claude (recommended)
    2. OPENAI_API_KEY -> OpenAI
    3. LLM_PROVIDER=mock -> Mock responses for testing
    
    Args:
        prompt: The prompt to send to the LLM
        provider: Force specific provider ('openai', 'anthropic', 'mock')
        model: Specific model name (optional, uses provider defaults)
    
    Returns:
        The raw text response from the LLM
    """
    # Auto-detect provider if not specified
    if provider is None:
        provider = os.getenv("LLM_PROVIDER", "").lower()
        
        if not provider:
            if os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
            elif os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "mock"
    
    if provider == "anthropic":
        return _call_anthropic(prompt, model or "claude-3-5-haiku-20241022")
    elif provider == "openai":
        return _call_openai(prompt, model or "gpt-4o-mini")
    elif provider == "mock":
        return _call_mock(prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'anthropic', or 'mock'")


def _call_openai(prompt: str, model: str) -> str:
    """Call OpenAI API"""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Get one at: https://platform.openai.com/api-keys")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("OpenAI returned None/empty response")
    
    return content


def _call_anthropic(prompt: str, model: str) -> str:
    """Call Anthropic Claude API"""
    from anthropic import Anthropic
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set. Get one at: https://console.anthropic.com/")
    
    client = Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        temperature=0.8,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def _call_mock(prompt: str) -> str:
    """Mock LLM for testing - returns dummy JSON responses"""
    import json
    
    print("⚠️  Using MOCK LLM - responses are dummy data for testing")
    
    # Detect what type of prompt this is (check most specific first)
    if "summarize and extract" in prompt.lower():
        return json.dumps({
            "chapter": 1,
            "summary_short": "Mock chapter summary for testing purposes.",
            "summary_detailed": "This is a longer mock summary with more details about what happened in the chapter for testing.",
            "new_facts": ["Mock fact 1", "Mock fact 2"]
        }, indent=2)
    
    elif "creating a volume outline" in prompt.lower():
        return json.dumps({
            "volume_title": "Debugging Reality - Volume 1",
            "chapters": [
                {
                    "number": i,
                    "title": f"Chapter {i}: Debug Session",
                    "goal": f"Introduce key mechanic {i}",
                    "conflict": "A new challenge appears",
                    "twist": "Something unexpected happens",
                    "hook": "Cliffhanger ending that leads to next chapter",
                    "pov": "Kai Winters"
                }
                for i in range(1, 11)
            ]
        }, indent=2)
    
    elif "write chapter" in prompt.lower():
        # Chapter text (not JSON)
        return """# Chapter 1: Reboot

The fluorescent lights buzzed overhead as Kai Winters stared at his monitor, his third energy drink of the night growing warm beside his keyboard. Another production bug. Another weekend sacrificed.

"Just one more deploy," he muttered, the mantra of the perpetually overworked developer.

He hit Enter. The screen flashed red.

And then everything went white.

---

When Kai opened his eyes, he wasn't in his cubicle anymore. He lay in a meadow, actual grass beneath his fingers, actual sunshine warming his face. His body felt... different. Lighter. Younger.

"What the—"

A translucent blue window materialized in front of his face, floating in midair like a badly positioned UI element.

**[SYSTEM INITIALIZED]**
**Welcome, Traveler Kai**
**Class: Debug Mage (Rare)**
**Level: 1**

Kai blinked. Then blinked again. He waved his hand through the window—it remained fixed in his vision, following his gaze.

"Oh no," he whispered. "I've been isekai'd."

---

The system interface was surprisingly intuitive, responding to his thoughts like a well-designed IDE. A skill tree branched before him, each node labeled with familiar terminology: *Loop Control*, *Conditional Casting*, *Exception Handling*.

This world's magic system was... code.

A rustling in the nearby bushes made him freeze. Something emerged—a small, glitching creature that looked like a corrupted sprite. Above its head, a label appeared:

**[Null Pointer Imp - Level 2]**

The imp's body flickered and stuttered, its movements jerky and unnatural. It locked eyes with Kai and hissed, a sound like static.

Kai's instincts kicked in. He focused on his skill list, finding a basic spell: *Print Attack*.

"Let's debug this," he said, and cast his first spell in this strange new world."""
    
    elif 'light novel "bible"' in prompt.lower():
        return json.dumps({
            "title": "CodeBound Chronicles",
            "genre": "Isekai fantasy / system",
            "premise": "A tired IT engineer is reborn in a world where spells compile like code, and bugs become monsters.",
            "setting": "A fantasy realm where magic follows programming logic, with kingdoms built on different coding paradigms.",
            "rules": [
                "Spells must be 'compiled' mentally before casting",
                "Syntax errors cause spell failure or corruption",
                "Bugs manifest as actual monsters based on error type",
                "Magic power scales with code optimization knowledge"
            ],
            "themes": ["Second chances", "Technical expertise in new context", "Logic vs chaos"],
            "main_characters": [
                {
                    "name": "Kai Winters",
                    "age": 28,
                    "role": "protagonist",
                    "goals": ["Master the magic system", "Find purpose in new world"],
                    "fears": ["Burnout repeating", "Losing his analytical edge"],
                    "personality": ["Logical", "Tired but determined", "Sarcastic"],
                    "speech_style": ["Technical metaphors", "Dry humor"],
                    "relationships": {}
                }
            ],
            "style_guide": ["Fast-paced", "Technical details balanced with action", "Light-hearted despite stakes"]
        }, indent=2)
    
    else:
        # Fallback - shouldn't reach here
        return '{"error": "Mock LLM could not detect prompt type"}'
