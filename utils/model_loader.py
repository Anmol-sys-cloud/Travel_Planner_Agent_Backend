import os
import requests
import socket
import urllib.parse
from requests.exceptions import RequestException
from dotenv import load_dotenv
from typing import Optional, Any
from pydantic import BaseModel, Field
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import logging


class HuggingFaceLLM:
    def __init__(self, model: str, api_key: str):
        self.model = model
        self.api_key = api_key

    def bind_tools(self, tools=None):
        return self

    def invoke(self, messages):
        prompt = self._build_prompt(messages)
        return self._call_huggingface(prompt)

    def _build_prompt(self, messages):
        if isinstance(messages, str):
            return messages
        if hasattr(messages, "content"):
            return getattr(messages, "content")
        if isinstance(messages, list):
            parts = []
            for item in messages:
                if hasattr(item, "content"):
                    parts.append(getattr(item, "content"))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(messages)

    def _call_huggingface(self, prompt: str):
        hf_base_url = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co")
        parsed_url = urllib.parse.urlparse(hf_base_url)
        if not parsed_url.scheme:
            raise RuntimeError(
                f"Invalid HUGGINGFACE_API_URL: '{hf_base_url}'. It must include http:// or https://."
            )

        url = f"{hf_base_url.rstrip('/')}/models/{self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "return_full_text": False,
            },
        }
        try:
            if parsed_url.hostname == "api-inference.huggingface.co":
                # quick DNS check for clearer error messaging
                try:
                    socket.gethostbyname(parsed_url.hostname)
                except socket.gaierror as e:
                    raise RuntimeError(
                        f"DNS resolution failed for {parsed_url.hostname}: {e}. "
                        "Ensure your machine has internet access and DNS can resolve the Hugging Face host, or set HUGGINGFACE_API_URL to a reachable endpoint."
                    )

            response = requests.post(url, headers=headers, json=payload, timeout=120)
        except RequestException as e:
            raise RuntimeError(
                f"Network request to Hugging Face failed: {e}. "
                "Check your internet connection, proxy settings, or firewall that may block the Hugging Face endpoint."
            )

        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face request failed {response.status_code}: {response.text}")

        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"Hugging Face returned non-JSON response: {response.text}")

        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Hugging Face error: {data['error']}")
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict) and "generated_text" in first:
                return first["generated_text"]
            if isinstance(first, str):
                return first

        raise RuntimeError(f"Unexpected Hugging Face response format: {data}")


class ConfigLoader:
    def __init__(self):
        print(f"Loaded config.....")
        self.config = load_config()
    
    def __getitem__(self, key):
        return self.config[key]


class ModelLoader(BaseModel):
    model_choice: str = "groq"
    config: Optional[ConfigLoader] = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.config = ConfigLoader()
    
    class Config:
        arbitrary_types_allowed = True
    
    def load_llm(self):
        """
        Load and return the LLM model.
        """
        # ensure environment variables from .env are loaded
        try:
            load_dotenv()
        except Exception:
            pass

        raw_choice = self.model_choice.strip()
        if "/" in raw_choice:
            provider, model_name = raw_choice.split("/", 1)
        else:
            provider = raw_choice
            try:
                model_name = self.config["llm"][provider]["model_name"]
            except KeyError:
                model_name = raw_choice

        logging.getLogger(__name__).info("LLM loading...")
        logging.getLogger(__name__).info(f"Loading model: {model_name} from provider: {provider}")
        
        if provider == "groq":
            print(f"Loading LLM from Groq for model {model_name}..............")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is not set. Set the GROQ_API_KEY environment variable or add it to a .env file."
                )
            masked = groq_api_key[:4] + "..." if len(groq_api_key) > 8 else "(set)"
            logging.getLogger(__name__).info(f"Using GROQ provider, GROQ_API_KEY present: {masked}")
            # compound-beta must be passed as the full string to ChatGroq
            llm = ChatGroq(model=model_name, api_key=groq_api_key)
            
        elif provider == "openai":
            print(f"Loading LLM from OpenAI for model {model_name}..............")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                print("OPENAI_API_KEY is not set. Routing through Groq as requested.")
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise RuntimeError("Both OPENAI_API_KEY and GROQ_API_KEY are missing.")
                # Pass the exact raw string (e.g. openai/gpt-oss-120b) to Groq
                llm = ChatGroq(model=raw_choice, api_key=groq_api_key)
            else:
                masked = openai_api_key[:4] + "..." if len(openai_api_key) > 8 else "(set)"
                logging.getLogger(__name__).info(f"Using OpenAI provider, OPENAI_API_KEY present: {masked}")
                llm = ChatOpenAI(model_name=model_name, api_key=openai_api_key)
                
        # ── ANTHROPIC (DISABLED) ────────────────────────────────────────────────
        # To enable: add ANTHROPIC_API_KEY=<your-key> to your .env file,
        # then uncomment the block below and the option in st_app.py.
        # elif provider == "anthropic":
        #     print(f"Loading LLM from Anthropic for model {model_name}..............")
        #     anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        #     if not anthropic_api_key:
        #         raise RuntimeError("ANTHROPIC_API_KEY is not set.")
        #     from langchain_anthropic import ChatAnthropic
        #     llm = ChatAnthropic(model_name=model_name, api_key=anthropic_api_key)
        # ────────────────────────────────────────────────────────────────────────
            
        elif provider == "huggingface":
            print(f"Loading LLM from Hugging Face for model {model_name}..............")
            hf_api_key = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HF_TOKEN")
            if not hf_api_key:
                raise RuntimeError(
                    "HUGGINGFACE_API_KEY or HF_TOKEN is not set. Set the Hugging Face key in your environment or .env file."
                )
            logging.getLogger(__name__).info("Using Hugging Face provider, API key present")
            hf_llm = HuggingFaceLLM(model=model_name, api_key=hf_api_key)
            llm = self._validate_hf_connectivity(hf_llm)
        else:
            raise RuntimeError(
                f"Unknown model provider '{provider}'. Supported values are groq, openai, anthropic, huggingface."
            )

        return llm

    def _validate_hf_connectivity(self, hf_llm: HuggingFaceLLM):
        # Perform a short connectivity check to ensure HF API is reachable.
        try:
            result = hf_llm._call_huggingface("Connectivity check")
            if isinstance(result, str) and result.startswith("Error:"):
                # If HF is unreachable, fall back to groq.
                print(f"Hugging Face connectivity failed: {result}")
                groq_api_key = os.getenv("GROQ_API_KEY")
                if not groq_api_key:
                    raise RuntimeError(
                        "Hugging Face is unavailable and GROQ_API_KEY is not set. Set a valid GROQ_API_KEY."
                    )
                model_name = self.config["llm"]["groq"]["model_name"]
                print("Falling back to Groq due to Hugging Face connectivity failure.")
                self.model_choice = "groq"
                return ChatGroq(model=model_name, api_key=groq_api_key)
        except Exception as e:
            print(f"Hugging Face connectivity validation exception: {e}")
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise
            model_name = self.config["llm"]["groq"]["model_name"]
            self.model_choice = "groq"
            return ChatGroq(model=model_name, api_key=groq_api_key)

        return hf_llm
    

    