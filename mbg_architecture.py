
"""
Masking-Balancing-Generation (MBG) pipeline.

Architecture:
[mask] -> [balance] -> [generate] -> END

Goal:
Reduce demographic representation bias before summarization.
"""
import re
from typing import TypedDict
from unittest import result
from langgraph.graph import StateGraph, END
from normalization import normalize_feedback_entries
from token_utils import empty_token_usage, add_ollama_token_usage

from config import (
    LLM_PROVIDER,
    LLM_MODEL_OPENAI,
    LLM_MODEL_OLLAMA,
    LLM_MODEL_GEMINI,
    LLM_MODEL_LOCAL_OLLAMA,
    TEMPERATURE,
    MAX_TOKENS,
)

from prompts import (
    MASKING_PROMPT,
    BALANCING_PROMPT,
    MBG_GENERATOR_PROMPT,
)

# ---------- MODEL SETUP ----------

USE_HEURISTIC_NORMALIZATION = True   # whether to apply heuristic normalization between masking and balancing, or let balancing handle it all

if LLM_PROVIDER == "openai":
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=LLM_MODEL_OPENAI,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

elif LLM_PROVIDER == "gemini":
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_GEMINI,
        temperature=TEMPERATURE,
    )

elif LLM_PROVIDER in ["ollama", "ollama-7b-local"]:
    from langchain_ollama import ChatOllama  

    model_name = (
        LLM_MODEL_LOCAL_OLLAMA
        if LLM_PROVIDER == "ollama-7b-local"
        else LLM_MODEL_OLLAMA
    )

    llm = ChatOllama(
        model=model_name,
        temperature=TEMPERATURE,
        num_predict=4096
    )

else:
    raise ValueError(f"Unknown provider: {LLM_PROVIDER}")


# ---------- STATE ----------

class MBGState(TypedDict):

    input_data: str
    user_prompt: str

    masked_data: str
    balanced_data: str

    summary: str


# ---------- MASK NODE ----------

def mask(state: MBGState):

    prompt = MASKING_PROMPT.format(
        input_data=state["input_data"]
    )

    response = llm.invoke(prompt)

    print("Masking complete")

    return {
        "masked_data": response.content
    }


# ---------- BALANCE NODE ----------

def balance(state: MBGState):

    prompt = BALANCING_PROMPT.format(
        masked_data=state["masked_data"]
    )

    response = llm.invoke(prompt)

    print("Balancing complete")

    return {
        "balanced_data": response.content
    }


# ---------- GENERATOR NODE ----------

def generate(state: MBGState):

    prompt = MBG_GENERATOR_PROMPT.format(
        user_prompt=state["user_prompt"],
        input_data=state["balanced_data"],
    )

    response = llm.invoke(prompt)

    print("Generation complete")

    return {
        "summary": response.content
    }


# ---------- GRAPH ----------

# chunking
def split_into_chunks(input_data: str, chunk_size: int = 10) -> list[str]:
    """
    Splits formatted feedback into chunks by APPLICATION NO.

    This version keeps multi-line feedback entries together.
    """

    matches = list(
        re.finditer(
            r"APPLICATION NO\.::\s*\d+",
            input_data
        )
    )

    entries = []

    for i, match in enumerate(matches):

        start = match.start()

        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = len(input_data)

        entry = input_data[start:end].strip()
        entries.append(entry)

    chunks = []

    for i in range(0, len(entries), chunk_size):
        chunk = "\n".join(entries[i:i + chunk_size])
        chunks.append(chunk)

    return chunks

def run_mbg(
    input_data: str,
    user_prompt: str,
):
    chunks = split_into_chunks(input_data, chunk_size=10)

    # of LLM calls = (mask + balance) * number of chunks + 1 generator call at the end
    n_requests = (2 * len(chunks)) + 1

    masked_chunks = []
    normalized_chunks = []
    balanced_chunks = []
    token_usage = empty_token_usage()

    print(f"Running MBG in {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks, start=1):

        print(f"\n--- MBG chunk {i}/{len(chunks)} ---")

        # 1. Masking
        mask_prompt = MASKING_PROMPT.format(
            input_data=chunk
        )

        mask_response = llm.invoke(mask_prompt)
        token_usage = add_ollama_token_usage(token_usage, mask_response)
        masked_chunk = mask_response.content
        masked_chunks.append(masked_chunk)

        print(f"Masking complete for chunk {i}")

        # 2. Heuristic normalization
        if USE_HEURISTIC_NORMALIZATION:
            normalized_chunk = normalize_feedback_entries(masked_chunk)
        else:
            normalized_chunk = masked_chunk

        normalized_chunks.append(normalized_chunk)

        print(f"Normalization complete for chunk {i}")

        # 3. Balancing
        balance_prompt = BALANCING_PROMPT.format(
            masked_data=normalized_chunk
        )

        balance_response = llm.invoke(balance_prompt)
        token_usage = add_ollama_token_usage(token_usage, balance_response)
        balanced_chunk = balance_response.content
        balanced_chunks.append(balanced_chunk)

        print(f"Balancing complete for chunk {i}")

    masked_data = "\n\n".join(masked_chunks)
    normalized_data = "\n\n".join(normalized_chunks)
    balanced_data = "\n\n".join(balanced_chunks)

    # 4. Generator
    generator_prompt = MBG_GENERATOR_PROMPT.format(
        user_prompt=user_prompt,
        input_data=balanced_data,
    )

    generator_response = llm.invoke(generator_prompt)
    token_usage = add_ollama_token_usage(token_usage, generator_response)
    summary = generator_response.content

    print("Generation complete")

    result = {
        "input_data": input_data,
        "user_prompt": user_prompt,
        "masked_data": masked_data,
        "normalized_data": normalized_data,
        "balanced_data": balanced_data,
        "summary": summary,
        "token_usage": token_usage,
        "n_requests": n_requests,
    }

    


