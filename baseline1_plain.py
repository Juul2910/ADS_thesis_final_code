"""
Single-node graph that takes input data and user prompt, calls LLM once, returns summary
[generate] -> END
"""
# -----------------------------------------------------------------------------------------------------------------
### Imports ###
import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI # for RC model
from langchain_google_genai import ChatGoogleGenerativeAI # for testing, free gemini
from langchain_ollama import ChatOllama # for Ollama model

from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from config import LLM_PROVIDER, LLM_MODEL_GEMINI, LLM_MODEL_OLLAMA, LLM_MODEL_OPENAI, LLM_MODEL_LOCAL_OLLAMA, TEMPERATURE, MAX_TOKENS
from config import OPENAI_BASE_URL
from prompts import SYSTEM_PROMPT, USER_PROMPT_1

#CHIARA toegevoegd:
import torch
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from transformers import AutoTokenizer, AutoModelForCausalLM
from token_utils import empty_token_usage, add_ollama_token_usage

load_dotenv() # Load .env, containing API keys, before LLM is initialised


# -----------------------------------------------------------------------------------------------------------------
### State ###

class Baseline1State(TypedDict):
    input_data: str     # raw input data as single string
    user_prompt: str    # user question
    summary: str        # output of generate node
    token_usage: dict   # token usage of the LLM call, stored as dict for extensibility (e.g. different fields for different providers)

# -----------------------------------------------------------------------------------------------------------------
### HF  #### #CHIARA toegevoegd: class for using HuggingFace Llama models, based on code from test_llama.py

HF_TOKENIZER = None
HF_MODEL = None

class HuggingFaceLlamaChat(BaseChatModel):
    model_id: str = Field(default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    max_new_tokens: int = Field(default=4096)
    temperature: float = Field(default=0.0)


    @property
    def _llm_type(self) -> str:
        return "hf_llama"

    def _load_model(self):
        global HF_TOKENIZER, HF_MODEL
        
        if HF_TOKENIZER is None:
            HF_TOKENIZER = AutoTokenizer.from_pretrained(self.model_id)
            HF_TOKENIZER.pad_token = HF_TOKENIZER.eos_token

        if HF_MODEL is None:
            HF_MODEL = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=torch.float16,
                device_map="auto",
            )

        return HF_TOKENIZER, HF_MODEL

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        tokenizer, model = self._load_model()

        hf_messages = []
        for m in messages:
            role = "assistant" if m.type == "ai" else "user"
            if m.type == "system":
                role = "system"
            hf_messages.append({"role": role, "content": m.content})

        encoded = tokenizer.apply_chat_template(
            hf_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.eos_token_id,
        }

        if self.temperature and self.temperature > 0:
            gen_kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            })
        else:
            gen_kwargs.update({
                "do_sample": False,
            })

        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=min(self.max_new_tokens, 300),
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][encoded["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

# -------------------------------------------------------------------------
### LLM ###

# Checks which model is put in LLM_PROVIDER in config.py, uses that model

if LLM_PROVIDER == "gemini":
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_GEMINI,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
    )
elif LLM_PROVIDER == "ollama":
    llm = ChatOllama(
        model=LLM_MODEL_OLLAMA,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,   # Ollama's equivalent of max_tokens
    )
elif LLM_PROVIDER == "openai":
    llm = ChatOpenAI(
        model=LLM_MODEL_OPENAI,
        base_url=OPENAI_BASE_URL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
elif LLM_PROVIDER == "ollama-local":
    llm = ChatOpenAI(
        model=LLM_MODEL_LOCAL_OLLAMA,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        base_url="http://localhost:11434/v1",
        api_key = "ollama",
    )
elif LLM_PROVIDER == "hf_llama": #CHIARA toegevoegd: option for HuggingFace Llama models, using the HuggingFaceLlamaChat class defined above
    llm = HuggingFaceLlamaChat(
        max_new_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )    
else:
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' in config.py") # raises error if provider not matched

# -----------------------------------------------------------------------------------------------------------------
### Node ###

def generate(state: Baseline1State) -> dict:
    """
    Single generation node.
    Calls LLM and stores summary.
    Receives current state and returns dict with updated fields.
    """
    system_prompt = SYSTEM_PROMPT.format(input_data=state["input_data"]) # insert input data into system prompt template

    messages = [
        SystemMessage(content=system_prompt),       # set model's role and grounding instructions
        HumanMessage(content=state["user_prompt"]), # human turn: specific question to answer
    ]

    # Send message to LLM
    token_usage = empty_token_usage()

    response = llm.invoke(messages)
    token_usage = add_ollama_token_usage(token_usage, response)

    summary = response.content

    return {
        "summary": summary,
        "token_usage": token_usage,
    }



# -----------------------------------------------------------------------------------------------------------------
### Graph ###

def build_graph():
    """
    Builds and compiles Baseline 1 graph.
    """
    builder = StateGraph(Baseline1State)

    builder.add_node("generate", generate)

    builder.set_entry_point("generate") # first node to execute
    builder.add_edge("generate", END)

    return builder.compile()

# -----------------------------------------------------------------------------------------------------------------
### Run function ###

def run_baseline1(input_data:str, user_prompt: str = USER_PROMPT_1) -> dict:
    """
    Run Baseline 1 on single output.
    Returns final state dict containing "summary".
    Wraps everything into single callable function.
    """
    graph = build_graph() # new graph for this run

    initial_state: Baseline1State = { # starting state
        "input_data": input_data,
        "user_prompt": user_prompt,
        "summary": "",
        "token_usage": empty_token_usage(),
    }

    # Run graph from entry point to END:
    result = graph.invoke(initial_state)
    
    return result

# -----------------------------------------------------------------------------------------------------------------
