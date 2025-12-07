# Local LLM Build Pipeline Research: LangGraph + Python + LM Studio

## Executive Summary

This document provides comprehensive research on designing and implementing a **fully local, Python-based, LangGraph-orchestrated pipeline** for building and deploying LLM agents. The pipeline is compatible with LM Studio's OpenAI-compatible API and produces deployable artifacts (`.ftpack`) without requiring cloud infrastructure.

**Key Findings:**
- ✅ **Feasible**: All components (LangGraph, LM Studio, Python tooling) support local-first workflows
- ⚠️ **Fine-tuning caveat**: LM Studio focuses on inference; actual fine-tuning requires external tools (Unsloth, Axolotl)
- ✅ **OpenAI API compatibility**: Drop-in replacement enables existing tooling to work locally
- ⚠️ **Hardware requirements**: Minimum 16GB RAM, GPU recommended for production workflows

---

## Table of Contents

1. [Architecture & Pipeline Design](#1-architecture--pipeline-design)
2. [Integration with Python](#2-integration-with-python)
3. [LM Studio Specifics](#3-lm-studio-specifics)
4. [Artifact & Deployment](#4-artifact--deployment)
5. [Existing Examples & Precedents](#5-existing-examples--precedents)
6. [Recommendations & Deliverables](#6-recommendations--deliverables)
7. [Limitations & Pitfalls](#7-limitations--pitfalls)

---

## 1. Architecture & Pipeline Design

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Local LLM Build Pipeline                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   Context   │ → │  Template   │ → │  Synthetic  │ → │   Linting   │     │
│  │ Collection  │   │  Expansion  │   │ Generation  │   │  & Filter   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│         ↓                                                      ↓            │
│  ┌─────────────┐                                      ┌─────────────┐      │
│  │   cop.yaml  │                                      │   Quality   │      │
│  │   Prompts   │                                      │   Scores    │      │
│  │   Personas  │                                      └─────────────┘      │
│  └─────────────┘                                               ↓            │
│                                                       ┌─────────────┐      │
│                                                       │   Dataset   │      │
│                                                       │   (.jsonl)  │      │
│                                                       └─────────────┘      │
│                                                                ↓            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        LangGraph Orchestration                       │   │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐             │   │
│  │  │ Config  │ → │ Metadata│ → │  Pack   │ → │ .ftpack │             │   │
│  │  │ Create  │   │  Create │   │  age    │   │ Artifact│             │   │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         LM Studio Server                             │   │
│  │         http://localhost:1234/v1 (OpenAI-compatible)                 │   │
│  │         ┌─────────────────────────────────────────┐                 │   │
│  │         │  Local GGUF Model (Llama, Mistral, etc) │                 │   │
│  │         └─────────────────────────────────────────┘                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Pipeline Stages

The pipeline consists of 8 sequential stages, designed for COP package building:

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **1. Context Collection** | Load cop.yaml, prompts, personas, guardrails | COP package files | Parsed context structure |
| **2. Template Expansion** | Resolve `{{variables}}` in prompts | Context + variables | Compiled prompts |
| **3. Synthetic Generation** | Generate training examples via local LLM | Compiled prompts | Synthetic completions |
| **4. Linting & Filtering** | Quality checks, deduplication, validation | Raw completions | Filtered dataset |
| **5. Dataset Generation** | Format as JSONL for fine-tuning | Filtered data | `train.jsonl` |
| **6. Config Creation** | Generate training configs | Dataset + params | `config.yaml` |
| **7. Metadata Creation** | Build manifest with checksums | All artifacts | `manifest.json` |
| **8. Packaging** | Bundle into deployable artifact | All files | `.ftpack` archive |

### 1.3 LangGraph DAG Structure

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
from dataclasses import dataclass

# Define state schema
class PipelineState(TypedDict):
    # Input
    cop_manifest_path: str
    variables: Dict[str, Any]
    
    # Stage outputs
    context: Dict[str, Any]
    compiled_prompts: List[str]
    synthetic_completions: List[Dict]
    filtered_data: List[Dict]
    dataset_path: str
    config: Dict[str, Any]
    metadata: Dict[str, Any]
    artifact_path: str
    
    # Status
    errors: List[str]
    warnings: List[str]
    stage: str

# Create graph
workflow = StateGraph(PipelineState)

# Add nodes (stages)
workflow.add_node("collect_context", collect_context_node)
workflow.add_node("expand_templates", expand_templates_node)
workflow.add_node("generate_synthetic", generate_synthetic_node)
workflow.add_node("lint_and_filter", lint_and_filter_node)
workflow.add_node("generate_dataset", generate_dataset_node)
workflow.add_node("create_config", create_config_node)
workflow.add_node("create_metadata", create_metadata_node)
workflow.add_node("package_artifact", package_artifact_node)

# Define edges (flow)
workflow.set_entry_point("collect_context")
workflow.add_edge("collect_context", "expand_templates")
workflow.add_edge("expand_templates", "generate_synthetic")
workflow.add_edge("generate_synthetic", "lint_and_filter")
workflow.add_edge("lint_and_filter", "generate_dataset")
workflow.add_edge("generate_dataset", "create_config")
workflow.add_edge("create_config", "create_metadata")
workflow.add_edge("create_metadata", "package_artifact")
workflow.add_edge("package_artifact", END)

# Compile
app = workflow.compile()
```

### 1.4 DAG Visualization

```
                    ┌──────────────────┐
                    │  START (Entry)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ collect_context  │
                    │                  │
                    │ Load cop.yaml,   │
                    │ prompts, etc.    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ expand_templates │
                    │                  │
                    │ Resolve          │
                    │ {{variables}}    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │generate_synthetic│
                    │                  │
                    │ Call LM Studio   │
                    │ for completions  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ lint_and_filter  │
                    │                  │
                    │ Quality checks,  │
                    │ deduplication    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ generate_dataset │
                    │                  │
                    │ Create .jsonl    │
                    │ for fine-tuning  │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  create_config   │
                    │                  │
                    │ Training params, │
                    │ LoRA config      │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ create_metadata  │
                    │                  │
                    │ Checksums,       │
                    │ manifest.json    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ package_artifact │
                    │                  │
                    │ Bundle .ftpack   │
                    │ archive          │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │       END        │
                    └──────────────────┘
```

---

## 2. Integration with Python

### 2.1 Implementing LangGraph Nodes

Each pipeline stage is implemented as a Python function that operates on the shared state:

```python
import yaml
import json
from pathlib import Path
from openai import OpenAI
from typing import Dict, List, Any
import hashlib

# Initialize LM Studio client
lm_studio = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"  # LM Studio doesn't require auth
)

def collect_context_node(state: PipelineState) -> PipelineState:
    """Stage 1: Load and parse COP package files."""
    manifest_path = Path(state["cop_manifest_path"])
    base_dir = manifest_path.parent
    
    # Load cop.yaml
    with open(manifest_path) as f:
        cop_manifest = yaml.safe_load(f)
    
    context = {
        "manifest": cop_manifest,
        "prompts": {},
        "personas": {},
        "guardrails": {},
        "knowledge": {},
        "tools": {}
    }
    
    # Load system prompt
    if "context" in cop_manifest and "system" in cop_manifest["context"]:
        prompt_path = base_dir / cop_manifest["context"]["system"]["source"]
        with open(prompt_path) as f:
            context["prompts"]["system"] = f.read()
    
    # Load personas
    if "personas" in cop_manifest.get("context", {}):
        for persona_ref in cop_manifest["context"]["personas"].get("available", []):
            persona_path = base_dir / persona_ref["source"]
            with open(persona_path) as f:
                context["personas"][persona_ref["name"]] = yaml.safe_load(f)
    
    # Load guardrails
    if "guardrails" in cop_manifest.get("context", {}):
        for gr_ref in cop_manifest["context"]["guardrails"]:
            gr_path = base_dir / gr_ref["source"]
            with open(gr_path) as f:
                context["guardrails"][gr_ref["name"]] = yaml.safe_load(f)
    
    return {**state, "context": context, "stage": "context_collected"}


def expand_templates_node(state: PipelineState) -> PipelineState:
    """Stage 2: Resolve template variables in prompts."""
    import re
    
    context = state["context"]
    variables = state.get("variables", {})
    compiled_prompts = []
    
    def resolve_variables(text: str, vars: Dict[str, Any]) -> str:
        """Replace {{variable}} with values."""
        pattern = r'\{\{(\w+)(?:\|([^}]+))?\}\}'
        
        def replacer(match):
            var_name = match.group(1)
            default = match.group(2)
            return str(vars.get(var_name, default or f"{{{{UNDEFINED:{var_name}}}}}"))
        
        return re.sub(pattern, replacer, text)
    
    # Compile system prompt
    if "system" in context["prompts"]:
        compiled = resolve_variables(context["prompts"]["system"], variables)
        compiled_prompts.append({
            "type": "system",
            "content": compiled,
            "source": "prompts/system.md"
        })
    
    return {**state, "compiled_prompts": compiled_prompts, "stage": "templates_expanded"}


def generate_synthetic_node(state: PipelineState) -> PipelineState:
    """Stage 3: Generate synthetic completions using local LLM."""
    compiled_prompts = state["compiled_prompts"]
    synthetic_completions = []
    
    system_prompt = next(
        (p["content"] for p in compiled_prompts if p["type"] == "system"),
        "You are a helpful assistant."
    )
    
    # Define example scenarios for synthetic data generation
    scenarios = [
        "Help me understand my order status",
        "I need a refund for my purchase",
        "What products do you offer?",
        "How do I contact support?",
        "Can you explain your return policy?"
    ]
    
    for scenario in scenarios:
        try:
            response = lm_studio.chat.completions.create(
                model="local-model",  # LM Studio uses loaded model
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": scenario}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            synthetic_completions.append({
                "system": system_prompt,
                "user": scenario,
                "assistant": response.choices[0].message.content,
                "model": response.model,
                "tokens": response.usage.total_tokens if response.usage else 0
            })
        except Exception as e:
            state["warnings"].append(f"Failed to generate for scenario '{scenario}': {e}")
    
    return {**state, "synthetic_completions": synthetic_completions, "stage": "synthetic_generated"}


def lint_and_filter_node(state: PipelineState) -> PipelineState:
    """Stage 4: Quality checks and filtering."""
    completions = state["synthetic_completions"]
    filtered = []
    
    for item in completions:
        # Quality checks
        assistant_text = item.get("assistant", "")
        
        # Check minimum length
        if len(assistant_text) < 50:
            state["warnings"].append(f"Filtered: Response too short ({len(assistant_text)} chars)")
            continue
        
        # Check for empty or error responses
        if not assistant_text.strip() or assistant_text.startswith("Error:"):
            state["warnings"].append("Filtered: Empty or error response")
            continue
        
        # Check for hallucination markers (basic)
        hallucination_markers = ["I don't have access", "I cannot", "As an AI"]
        if any(marker.lower() in assistant_text.lower() for marker in hallucination_markers):
            # Flag but don't filter - might be valid safety responses
            item["flagged"] = True
        
        # Deduplicate by content hash
        content_hash = hashlib.md5(assistant_text.encode()).hexdigest()
        item["content_hash"] = content_hash
        
        filtered.append(item)
    
    # Remove exact duplicates
    seen_hashes = set()
    unique_filtered = []
    for item in filtered:
        if item["content_hash"] not in seen_hashes:
            seen_hashes.add(item["content_hash"])
            unique_filtered.append(item)
    
    return {**state, "filtered_data": unique_filtered, "stage": "filtered"}


def generate_dataset_node(state: PipelineState) -> PipelineState:
    """Stage 5: Generate JSONL dataset for fine-tuning."""
    filtered_data = state["filtered_data"]
    output_dir = Path("build/datasets")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_path = output_dir / "train.jsonl"
    
    with open(dataset_path, "w") as f:
        for item in filtered_data:
            # Format for fine-tuning (OpenAI/Alpaca format)
            entry = {
                "messages": [
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["user"]},
                    {"role": "assistant", "content": item["assistant"]}
                ]
            }
            f.write(json.dumps(entry) + "\n")
    
    return {**state, "dataset_path": str(dataset_path), "stage": "dataset_generated"}


def create_config_node(state: PipelineState) -> PipelineState:
    """Stage 6: Create training/deployment configuration."""
    context = state["context"]
    dataset_path = state["dataset_path"]
    
    config = {
        "base_model": "meta-llama/Llama-3.2-3B-Instruct",
        "dataset": {
            "path": dataset_path,
            "format": "messages"
        },
        "training": {
            "method": "lora",
            "lora_r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "learning_rate": 2e-4,
            "epochs": 3,
            "batch_size": 4,
            "gradient_accumulation_steps": 4
        },
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": "float16"
        },
        "output": {
            "model_name": context["manifest"].get("name", "custom-agent"),
            "merge_adapter": True,
            "export_gguf": True,
            "gguf_quantization": "Q4_K_M"
        }
    }
    
    # Write config
    config_path = Path("build/config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return {**state, "config": config, "stage": "config_created"}


def create_metadata_node(state: PipelineState) -> PipelineState:
    """Stage 7: Generate manifest with checksums and metadata."""
    import datetime
    
    context = state["context"]
    dataset_path = state["dataset_path"]
    
    # Calculate checksums
    def file_checksum(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    metadata = {
        "version": "1.0.0",
        "name": context["manifest"].get("name", "custom-agent"),
        "description": context["manifest"].get("description", ""),
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "0.1.0",
        "files": {
            "dataset": {
                "path": dataset_path,
                "checksum": file_checksum(dataset_path),
                "entries": len(state["filtered_data"])
            },
            "config": {
                "path": "build/config.yaml",
                "checksum": file_checksum("build/config.yaml")
            }
        },
        "statistics": {
            "total_synthetic": len(state.get("synthetic_completions", [])),
            "filtered_count": len(state.get("filtered_data", [])),
            "warnings_count": len(state.get("warnings", []))
        }
    }
    
    # Write metadata
    metadata_path = Path("build/manifest.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    return {**state, "metadata": metadata, "stage": "metadata_created"}


def package_artifact_node(state: PipelineState) -> PipelineState:
    """Stage 8: Bundle everything into .ftpack archive."""
    import tarfile
    import shutil
    
    context = state["context"]
    package_name = context["manifest"].get("name", "custom-agent")
    version = state["metadata"].get("version", "1.0.0")
    
    artifact_name = f"{package_name}-{version}.ftpack"
    artifact_path = Path("dist") / artifact_name
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tarball
    with tarfile.open(artifact_path, "w:gz") as tar:
        # Add dataset
        tar.add(state["dataset_path"], arcname="data/train.jsonl")
        
        # Add config
        tar.add("build/config.yaml", arcname="config.yaml")
        
        # Add manifest
        tar.add("build/manifest.json", arcname="manifest.json")
        
        # Add original COP manifest
        tar.add(state["cop_manifest_path"], arcname="cop.yaml")
    
    return {**state, "artifact_path": str(artifact_path), "stage": "packaged"}
```

### 2.2 Best Practices for Local LLM Calls

#### Connection Configuration

```python
from openai import OpenAI
import httpx

def create_lm_studio_client(
    base_url: str = "http://localhost:1234/v1",
    timeout: float = 120.0,
    max_retries: int = 3
) -> OpenAI:
    """Create configured LM Studio client with timeouts and retries."""
    return OpenAI(
        base_url=base_url,
        api_key="not-needed",
        timeout=httpx.Timeout(timeout, connect=10.0),
        max_retries=max_retries
    )

# Usage
client = create_lm_studio_client()
```

#### Handling Caching

```python
import hashlib
import json
from pathlib import Path
from functools import lru_cache
from typing import Optional
import diskcache

# Initialize disk cache
cache = diskcache.Cache("./cache/llm_responses")

def cached_completion(
    client: OpenAI,
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 500,
    cache_ttl: int = 3600  # 1 hour
) -> dict:
    """Get completion with caching."""
    # Create cache key from request
    cache_key = hashlib.sha256(
        json.dumps({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }, sort_keys=True).encode()
    ).hexdigest()
    
    # Check cache
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    
    # Make request
    response = client.chat.completions.create(
        model="local-model",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    result = {
        "content": response.choices[0].message.content,
        "model": response.model,
        "usage": response.usage.model_dump() if response.usage else None
    }
    
    # Cache result
    cache.set(cache_key, result, expire=cache_ttl)
    
    return result
```

#### Parallelism with Rate Limiting

```python
import asyncio
from asyncio import Semaphore
from typing import List, Dict, Any
from openai import AsyncOpenAI

async_client = AsyncOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed"
)

async def parallel_completions(
    prompts: List[Dict[str, Any]],
    max_concurrent: int = 3,  # Limit for local inference
    delay_between: float = 0.5  # Prevent overwhelming local server
) -> List[Dict]:
    """Run completions in parallel with rate limiting."""
    semaphore = Semaphore(max_concurrent)
    results = []
    
    async def process_one(prompt_data: Dict) -> Dict:
        async with semaphore:
            try:
                response = await async_client.chat.completions.create(
                    model="local-model",
                    messages=prompt_data["messages"],
                    temperature=prompt_data.get("temperature", 0.7),
                    max_tokens=prompt_data.get("max_tokens", 500)
                )
                await asyncio.sleep(delay_between)
                return {
                    "input": prompt_data,
                    "output": response.choices[0].message.content,
                    "success": True
                }
            except Exception as e:
                return {
                    "input": prompt_data,
                    "error": str(e),
                    "success": False
                }
    
    tasks = [process_one(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    
    return results
```

#### Reproducibility

```python
import random
import numpy as np
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class ReproducibilityConfig:
    """Configuration for reproducible runs."""
    seed: int = 42
    temperature: float = 0.0  # Deterministic
    top_p: float = 1.0
    model_name: str = ""
    model_hash: Optional[str] = None

def set_reproducibility(config: ReproducibilityConfig):
    """Set all random seeds for reproducibility."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    # Note: LLM inference is inherently non-deterministic even with temp=0
    # but this minimizes variation

def create_reproducibility_manifest(config: ReproducibilityConfig) -> dict:
    """Create manifest for reproducing the run."""
    return {
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "model": {
            "name": config.model_name,
            "hash": config.model_hash
        },
        "python_version": sys.version,
        "dependencies": {
            "langgraph": langgraph.__version__,
            "openai": openai.__version__
        }
    }
```

---

## 3. LM Studio Specifics

### 3.1 LM Studio as Local OpenAI-Compatible API

LM Studio provides a drop-in replacement for OpenAI's API, running on `localhost:1234/v1`.

#### Starting the Server

**GUI Method:**
1. Open LM Studio
2. Go to "Local Server" tab (or Developer tab)
3. Click "Start Server"
4. Server runs at `http://localhost:1234`

**CLI Method:**
```bash
# Using lms CLI
lms server start

# With specific model
lms server start --model "lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF"

# Check status
lms server status
```

#### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /v1/models` | List loaded models |
| `POST /v1/chat/completions` | Chat completion (OpenAI compatible) |
| `POST /v1/completions` | Text completion |
| `POST /v1/embeddings` | Generate embeddings |

#### Python Integration

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"  # Any string works
)

# List available models
models = client.models.list()
print([m.id for m in models.data])

# Chat completion
response = client.chat.completions.create(
    model="lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ],
    temperature=0.7,
    max_tokens=500,
    stream=False
)

print(response.choices[0].message.content)
```

### 3.2 Fine-Tuning & LoRA Support

**Important**: LM Studio (as of late 2024) **does not natively support fine-tuning**. It is primarily an inference server for pre-trained/fine-tuned models.

#### Fine-Tuning Workflow (External Tools)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    Fine-Tuning Workflow                               │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Generate dataset    →  Pipeline creates train.jsonl               │
│                                                                       │
│  2. Fine-tune with      →  Use Unsloth, Axolotl, or                  │
│     external tool          transformers + PEFT                        │
│                                                                       │
│  3. Export to GGUF      →  Convert to quantized GGUF format          │
│                                                                       │
│  4. Load in LM Studio   →  Drag & drop GGUF file                     │
│                                                                       │
│  5. Serve via API       →  http://localhost:1234/v1                  │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

#### Recommended Fine-Tuning Tools

**1. Unsloth (Recommended for efficiency)**
```python
from unsloth import FastLanguageModel
import torch

# Load base model with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Train with your dataset
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,  # Your JSONL dataset
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs",
    ),
)

trainer.train()

# Save and convert to GGUF
model.save_pretrained_gguf("model", tokenizer, quantization_method="q4_k_m")
```

**2. Axolotl (Full-featured)**
```yaml
# axolotl_config.yaml
base_model: meta-llama/Llama-3.2-3B-Instruct
model_type: LlamaForCausalLM

load_in_4bit: true
adapter: lora
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
  - k_proj
  - o_proj

datasets:
  - path: ./data/train.jsonl
    type: sharegpt

sequence_len: 2048
sample_packing: true

output_dir: ./outputs/lora-out
```

```bash
# Run training
accelerate launch -m axolotl.cli.train axolotl_config.yaml
```

### 3.3 GGUF Conversion for LM Studio

After fine-tuning, convert to GGUF for LM Studio:

```python
# Using llama.cpp conversion
# First, merge LoRA adapter with base model

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Load and merge LoRA
model = PeftModel.from_pretrained(base_model, "./outputs/lora-out")
model = model.merge_and_unload()

# Save merged model
model.save_pretrained("./merged_model")
tokenizer.save_pretrained("./merged_model")
```

```bash
# Convert to GGUF using llama.cpp
python llama.cpp/convert_hf_to_gguf.py ./merged_model \
    --outfile ./model.gguf \
    --outtype q4_k_m
```

### 3.4 Loading Custom Models in LM Studio

1. **Drag & Drop**: Simply drag the `.gguf` file into LM Studio
2. **CLI**: `lms load ./model.gguf`
3. **API**: Models are auto-discovered from `~/.cache/lm-studio/models/`

---

## 4. Artifact & Deployment

### 4.1 `.ftpack` Artifact Structure

```
my-agent-1.0.0.ftpack (tar.gz archive)
├── manifest.json           # Package metadata & checksums
├── cop.yaml               # Original COP manifest
├── config.yaml            # Training/deployment config
├── data/
│   ├── train.jsonl        # Fine-tuning dataset
│   └── eval.jsonl         # Evaluation dataset (optional)
├── model/                 # (Optional) Pre-trained model files
│   └── adapter/           # LoRA adapter weights
│       ├── adapter_config.json
│       └── adapter_model.safetensors
├── prompts/               # Compiled prompts
│   └── system.md
├── context/               # Full context bundle
│   └── bundle.json
└── checksums.sha256       # File integrity
```

### 4.2 Manifest Schema

```json
{
  "version": "1.0.0",
  "name": "customer-support-agent",
  "description": "Fine-tuned customer support agent",
  "created_at": "2024-12-07T10:30:00Z",
  "pipeline_version": "0.1.0",
  
  "base_model": {
    "name": "meta-llama/Llama-3.2-3B-Instruct",
    "type": "causal-lm",
    "quantization": "Q4_K_M"
  },
  
  "training": {
    "method": "lora",
    "epochs": 3,
    "dataset_size": 1000,
    "eval_loss": 0.45
  },
  
  "files": {
    "dataset": {
      "path": "data/train.jsonl",
      "checksum": "sha256:abc123...",
      "entries": 1000
    },
    "adapter": {
      "path": "model/adapter/",
      "checksum": "sha256:def456..."
    }
  },
  
  "compatibility": {
    "lm_studio": ">=0.3.0",
    "ollama": ">=0.2.0",
    "transformers": ">=4.40.0"
  }
}
```

### 4.3 Deployment Strategies

#### A. LM Studio Deployment

```python
import shutil
from pathlib import Path

def deploy_to_lm_studio(ftpack_path: str, lm_studio_models_dir: str = None):
    """Deploy .ftpack model to LM Studio."""
    import tarfile
    
    if lm_studio_models_dir is None:
        # Default LM Studio models directory
        lm_studio_models_dir = Path.home() / ".cache" / "lm-studio" / "models"
    
    lm_studio_models_dir = Path(lm_studio_models_dir)
    lm_studio_models_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract ftpack
    with tarfile.open(ftpack_path, "r:gz") as tar:
        # Read manifest
        manifest_file = tar.extractfile("manifest.json")
        manifest = json.load(manifest_file)
        
        model_name = manifest["name"]
        version = manifest["version"]
        
        # Create model directory
        model_dir = lm_studio_models_dir / f"{model_name}-{version}"
        model_dir.mkdir(exist_ok=True)
        
        # Extract model files
        for member in tar.getmembers():
            if member.name.startswith("model/"):
                tar.extract(member, model_dir)
    
    print(f"Deployed to: {model_dir}")
    return model_dir
```

#### B. LMS CLI Integration

```bash
#!/bin/bash
# deploy.sh - Deploy ftpack using lms CLI

FTPACK_PATH=$1
MODEL_NAME=$(tar -xzf "$FTPACK_PATH" -O manifest.json | jq -r '.name')

# Extract GGUF model
tar -xzf "$FTPACK_PATH" -C /tmp/ftpack_extract

# Load into LM Studio
lms load "/tmp/ftpack_extract/model/model.gguf" --name "$MODEL_NAME"

# Start server with model
lms server start --model "$MODEL_NAME"

echo "Model deployed and server started at http://localhost:1234"
```

#### C. Docker Deployment

```dockerfile
# Dockerfile for self-contained deployment
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install openai fastapi uvicorn

# Copy ftpack contents
COPY extracted_ftpack/ /app/model/

# Copy runtime server
COPY server.py /app/

EXPOSE 8080

CMD ["python", "server.py"]
```

```python
# server.py - Lightweight inference server
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI()

# Load context from ftpack
with open("/app/model/context/bundle.json") as f:
    context = json.load(f)

class ChatRequest(BaseModel):
    messages: list

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest):
    # Forward to LM Studio or local inference
    # This is a proxy that applies COP context
    pass
```

### 4.4 CI/CD Pipeline

```yaml
# .github/workflows/build-ftpack.yml
name: Build COP Package

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install langgraph openai pyyaml diskcache
      
      - name: Validate COP manifest
        run: |
          pip install -e .
          cop validate examples/customer-support-agent
      
      - name: Run build pipeline
        run: |
          python build_pipeline.py \
            --manifest examples/customer-support-agent/cop.yaml \
            --output dist/
        env:
          # For CI, use mock LLM or skip synthetic generation
          SKIP_SYNTHETIC: true
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: ftpack
          path: dist/*.ftpack

  # Optional: Integration test with local LLM
  test:
    needs: build
    runs-on: self-hosted  # Requires GPU runner
    
    steps:
      - name: Download ftpack
        uses: actions/download-artifact@v4
        with:
          name: ftpack
      
      - name: Deploy and test
        run: |
          # Start LM Studio server (pre-configured on runner)
          ./test_deployment.sh *.ftpack
```

---

## 5. Existing Examples & Precedents

### 5.1 LangGraph + Python Orchestration Projects

| Project | Description | Link |
|---------|-------------|------|
| **KnowledgeGraphQA-Langgraph** | Knowledge graph generation + QA with LangGraph | [GitHub](https://github.com/samitugal/KnowledgeGraphQA-Langgraph) |
| **LangGraph Examples** | Official LangGraph example repository | [GitHub](https://github.com/langchain-ai/langgraph/tree/main/examples) |
| **RAG with LangGraph** | Retrieval-augmented generation pipeline | [Tutorial](https://www.geeksforgeeks.org/artificial-intelligence/rag-system-with-langchain-and-langgraph/) |

### 5.2 Local LLM Fine-Tuning Pipelines

| Tool | Use Case | Link |
|------|----------|------|
| **Unsloth** | Efficient LoRA fine-tuning (2x faster) | [GitHub](https://github.com/unslothai/unsloth) |
| **Axolotl** | Full-featured fine-tuning framework | [GitHub](https://github.com/OpenAccess-AI-Collective/axolotl) |
| **LLaMA-Factory** | Easy fine-tuning with web UI | [GitHub](https://github.com/hiyouga/LLaMA-Factory) |
| **PEFT** | Parameter-efficient fine-tuning | [HuggingFace](https://github.com/huggingface/peft) |

### 5.3 LM Studio Integration Examples

```python
# Example: Using LM Studio with LangChain
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableSequence

# Configure for LM Studio
llm = ChatOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="not-needed",
    model="local-model",
    temperature=0.7
)

# Create chain
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{input}")
])

chain = prompt | llm

# Run
response = chain.invoke({"input": "Hello!"})
print(response.content)
```

---

## 6. Recommendations & Deliverables

### 6.1 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Recommended Local Pipeline                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────┐                                                 │
│  │   COP Package  │  cop.yaml, prompts/, personas/, etc.            │
│  └───────┬────────┘                                                 │
│          │                                                          │
│          ▼                                                          │
│  ┌────────────────┐                                                 │
│  │   LangGraph    │  Orchestrates all stages                        │
│  │   Pipeline     │  Handles state, caching, parallelism            │
│  └───────┬────────┘                                                 │
│          │                                                          │
│          ├──────────────────┐                                       │
│          │                  │                                       │
│          ▼                  ▼                                       │
│  ┌────────────────┐  ┌────────────────┐                            │
│  │   LM Studio    │  │   Unsloth/     │                            │
│  │   (Inference)  │  │   Axolotl      │                            │
│  │                │  │   (Training)   │                            │
│  └───────┬────────┘  └───────┬────────┘                            │
│          │                   │                                      │
│          └─────────┬─────────┘                                      │
│                    │                                                │
│                    ▼                                                │
│          ┌────────────────┐                                         │
│          │    .ftpack     │  Deployable artifact                    │
│          │    Artifact    │  Contains model + context               │
│          └────────────────┘                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Implementation Roadmap

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Phase 1: Foundation** | Set up LangGraph skeleton, LM Studio integration, basic pipeline | 1 week |
| **Phase 2: Synthetic Data** | Template expansion, synthetic generation, quality filtering | 1 week |
| **Phase 3: Fine-tuning** | Integrate Unsloth/Axolotl, GGUF conversion, testing | 2 weeks |
| **Phase 4: Packaging** | .ftpack format, deployment scripts, documentation | 1 week |
| **Phase 5: CI/CD** | GitHub Actions, automated testing, artifact publishing | 1 week |

### 6.3 Minimum Viable Pipeline

```python
# minimal_pipeline.py - Complete working example
from langgraph.graph import StateGraph, END
from openai import OpenAI
from typing import TypedDict
import yaml
import json

class State(TypedDict):
    manifest_path: str
    context: dict
    dataset: list
    output_path: str

def load_context(state: State) -> State:
    with open(state["manifest_path"]) as f:
        manifest = yaml.safe_load(f)
    return {**state, "context": manifest}

def generate_data(state: State) -> State:
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="x")
    dataset = []
    
    # Generate synthetic examples
    for i in range(10):
        resp = client.chat.completions.create(
            model="local-model",
            messages=[{"role": "user", "content": f"Example {i}"}]
        )
        dataset.append({"input": f"Example {i}", "output": resp.choices[0].message.content})
    
    return {**state, "dataset": dataset}

def save_output(state: State) -> State:
    with open("output.jsonl", "w") as f:
        for item in state["dataset"]:
            f.write(json.dumps(item) + "\n")
    return {**state, "output_path": "output.jsonl"}

# Build graph
workflow = StateGraph(State)
workflow.add_node("load", load_context)
workflow.add_node("generate", generate_data)
workflow.add_node("save", save_output)
workflow.set_entry_point("load")
workflow.add_edge("load", "generate")
workflow.add_edge("generate", "save")
workflow.add_edge("save", END)

app = workflow.compile()

# Run
result = app.invoke({"manifest_path": "cop.yaml"})
print(f"Output saved to: {result['output_path']}")
```

---

## 7. Limitations & Pitfalls

### 7.1 Hardware Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **RAM** | 16 GB | 32+ GB | For 7B parameter models |
| **VRAM** | 8 GB | 24+ GB | For fine-tuning; inference can use CPU |
| **Storage** | 50 GB | 200+ GB | Model weights, datasets, cache |
| **CPU** | 8 cores | 16+ cores | Parallelism in pipeline |

### 7.2 Performance Considerations

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **Local inference speed** | 5-20 tokens/sec on CPU | Use GPU, smaller models (3B), or quantization |
| **Memory pressure** | OOM during generation | Batch processing, streaming, offloading |
| **Pipeline latency** | Minutes per run | Caching, incremental builds, parallelism |
| **Model loading time** | 30-60 seconds | Keep server running, use preloaded models |

### 7.3 Reproducibility Challenges

- **LLM non-determinism**: Even with `temperature=0`, outputs vary slightly
- **Model version drift**: Same model name may have different weights over time
- **Hardware differences**: CPU vs GPU inference can produce different results

**Mitigations:**
```python
# Pin model versions with checksums
model_config = {
    "name": "TheBloke/Llama-2-7B-Chat-GGUF",
    "file": "llama-2-7b-chat.Q4_K_M.gguf",
    "sha256": "abc123...",  # Verify on load
    "quantization": "Q4_K_M"
}

# Log all inference parameters
inference_log = {
    "timestamp": datetime.utcnow().isoformat(),
    "model": model_config,
    "temperature": 0.0,
    "seed": 42,
    "max_tokens": 500,
    "system_info": {
        "cpu": platform.processor(),
        "gpu": get_gpu_info(),
        "ram_gb": psutil.virtual_memory().total / (1024**3)
    }
}
```

### 7.4 Known Limitations

| Limitation | Description | Workaround |
|------------|-------------|------------|
| **No native fine-tuning in LM Studio** | LM Studio is inference-only | Use Unsloth/Axolotl externally |
| **Limited batch inference** | Sequential processing | Use async with semaphores |
| **No multi-GPU in LM Studio** | Single GPU only | Use vLLM or TGI for multi-GPU |
| **Context window limits** | Model-dependent (4K-128K) | Chunking, summarization |

### 7.5 Security Considerations

- **Local-only network**: Keep LM Studio server on `localhost` only
- **No authentication**: LM Studio doesn't require API keys (by design)
- **Model provenance**: Verify model checksums before loading
- **Data privacy**: All processing stays local, but audit prompts for PII

---

## Appendix A: Complete Pipeline Code

See the full implementation in the repository:
- `build/local_pipeline.py` - Main pipeline implementation
- `build/nodes/` - Individual pipeline nodes
- `build/utils/` - Helper functions
- `build/config.yaml` - Default configuration

## Appendix B: Tool Versions Tested

| Tool | Version | Notes |
|------|---------|-------|
| LangGraph | 0.2.x | Stable |
| LM Studio | 0.3.5+ | OpenAI compatibility |
| Python | 3.10-3.12 | Recommended 3.11 |
| Unsloth | 2024.11+ | For fine-tuning |
| llama.cpp | Latest | GGUF conversion |

---

*Document Version: 1.0.0*  
*Last Updated: December 2024*  
*Author: Research Team*

