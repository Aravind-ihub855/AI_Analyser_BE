import os 
import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq   
from langchain_mistralai import ChatMistralAI

from pathlib import Path

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

# --- CUSTOM AI (NGROK) ---
async def call_gptoss_ai(prompt: str) -> str:
    """Calls the custom AI service hosted on Ngrok."""
    url = "https://f54a-103-196-28-74.ngrok-free.app/chat"
    headers = {
        "x-api-key": "testing",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=30.0)
            if response.status_code == 200:
                result = response.json()
                return result.get("response", str(result))
            else:
                return f"Error: Custom AI service returned status {response.status_code}"
        except Exception as e:
            return f"Error: Unable to connect to custom AI service. {str(e)}"

# --- MISTRAL AI ---
def get_mistral_llm(api_key: str = None):
    """Configure Mistral LLM with hardcoded settings or provided key."""
    key_to_use = api_key if api_key else os.getenv("MISTRAL_API_KEY")
    if not key_to_use:
        raise ValueError("Mistral API key not set in environment.")
    return ChatMistralAI(
        model="mistral-medium-2505",
        api_key=key_to_use,
        temperature=0.7,
        max_tokens=32768,
    )

# --- GROQ AI ---
def get_groq_llm(model: str = "openai/gpt-oss-120b", temperature: float = 0, top_p: float = 1.0):
    """Returns a LangChain ChatGroq instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set in .env")
    return ChatGroq(
        groq_api_key=api_key,
        model=model,
        temperature=temperature,
        model_kwargs={"top_p": top_p}
    )

async def call_groq_ai(prompt: str, temperature: float = 0, top_p: float = 1.0) -> str:
    """Calls Groq Llama 3 AI."""
    try:
        llm = get_groq_llm(temperature=temperature, top_p=top_p)
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        return f"Error calling Groq: {str(e)}"

# --- GEMINI AI ---
async def call_gemini_ai(prompt: str) -> str:
    """Calls Google Gemini AI."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Error: GOOGLE_API_KEY not set in .env"
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )
        response = await llm.ainvoke(prompt)
        return response.content
    except Exception as e:
        return f"Error calling Gemini: {str(e)}"

# --- MISTRAL AI (CUSTOM CLOUD) ---
async def call_mistral_ai(prompt: str) -> str:
    """Calls the Mistral 7B Instruct service."""
    url = "https://ai.jayasimacloud.me/v1/chat/completions"
    api_key = os.getenv("MISTRAL_API_KEY") or "YOUR_API_KEY"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mistral",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=30.0)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: Mistral service returned status {response.status_code}. {response.text}"
        except Exception as e:
            return f"Error: Unable to connect to Mistral service. {str(e)}"

# --- OPENROUTER AI ---
async def call_openrouter_ai(prompt: str) -> str:
    """Calls OpenRouter with arcee-ai/trinity-large-preview:free model."""
    url = "https://openrouter.ai/api/v1/chat/completions"
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "arcee-ai/trinity-large-preview:free",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    async with httpx.AsyncClient() as client:
        try:
            # OpenRouter can sometimes be slow; setting timeout to 60s
            response = await client.post(url, headers=headers, json=payload, timeout=60.0)
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: OpenRouter service returned status {response.status_code}. {response.text}"
        except Exception as e:
            return f"Error: Unable to connect to OpenRouter service. {str(e)}"