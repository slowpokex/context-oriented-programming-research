"""
Synthetic Data Generation Pipeline using LangGraph

This module implements a DAG-based pipeline for generating synthetic
training data using a local LLM (LM Studio, Ollama, etc.)
"""

import json
import hashlib
import re
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# Try to import LangGraph (optional dependency)
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

from openai import OpenAI, APIError, APIConnectionError, RateLimitError, AuthenticationError

from .logger import PipelineLogger, APICallStats
from .constants import DEFAULT_LLM_ENDPOINT, DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL, DEFAULT_API_KEY


class PipelineState(TypedDict):
    """State passed between pipeline nodes."""
    # Input
    system_prompt: str
    personas: Dict[str, Any]
    guardrails: List[Dict]
    tools: List[Dict]
    config: Dict[str, Any]
    
    # Processing stages
    scenarios: List[str]
    expanded_scenarios: List[Dict]  # Enriched scenarios with context
    conversations: List[Dict]
    multi_turn_conversations: List[Dict]  # Multi-turn versions
    scored_conversations: List[Dict]  # With quality scores
    filtered_conversations: List[Dict]
    
    # Output
    dataset: List[Dict]
    stats: Dict[str, Any]
    errors: List[str]


# Reference descriptions for semantic agent type classification
AGENT_TYPE_DESCRIPTIONS = {
    "code_generation": """
        An AI assistant specialized in generating, reviewing, and explaining code.
        Helps developers write functions, classes, tests, APIs, and debug issues.
        Works with programming languages like Python, TypeScript, JavaScript, Rust, Go.
        Generates code blocks, explains syntax, follows best practices and conventions.
    """,
    "customer_support": """
        A customer service agent that handles inquiries, complaints, and support requests.
        Processes refunds, orders, account issues, and escalates complex problems.
        Empathetic, professional, follows company policies, uses tools to look up information.
        Helps customers resolve issues efficiently and ensures satisfaction.
    """,
    "educational": """
        A tutor or teacher that explains concepts, provides examples, and helps students learn.
        Patient, uses analogies and simple explanations, checks understanding.
        Breaks down complex topics, provides practice problems, encourages questions.
        Adapts explanations to student's level of understanding.
    """,
    "data_analysis": """
        An analyst that interprets data, creates reports, and generates business insights.
        Works with metrics, charts, dashboards, SQL queries, and statistical analysis.
        Identifies trends, anomalies, and patterns in data. Creates visualizations.
        Helps with data-driven decision making and reporting.
    """,
    "general": """
        A general-purpose AI assistant that helps with various tasks and questions.
        Informative, helpful, and adaptable to different types of requests.
        Can answer questions, provide information, help with tasks, and have conversations.
    """
}


@dataclass
class SyntheticDataPipeline:
    """LangGraph-based pipeline for generating synthetic training data."""
    
    endpoint: str = DEFAULT_LLM_ENDPOINT
    model: str = DEFAULT_LLM_MODEL
    api_key: str = DEFAULT_API_KEY
    temperature: float = 0.8
    max_tokens: int = 2048  # Increased for code generation
    num_samples: int = 10
    multi_turn_ratio: float = 0.3  # 30% of conversations are multi-turn
    console: Optional[Console] = None
    verbose: bool = False
    
    # Embedding-based classification
    use_embeddings: bool = True
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    
    # RAG integration
    vector_index: Optional[Any] = None  # VectorIndex for RAG retrieval
    rag_top_k: int = 5                   # Number of chunks to retrieve
    use_rag: bool = False                # Enable RAG-enhanced generation
    
    _client: Optional[OpenAI] = field(default=None, init=False, repr=False)
    _agent_type: Optional[str] = field(default=None, init=False, repr=False)
    _embedding_cache: Dict[str, List[float]] = field(default_factory=dict, init=False, repr=False)
    _ref_embeddings: Dict[str, List[float]] = field(default_factory=dict, init=False, repr=False)
    _logger: Optional[PipelineLogger] = field(default=None, init=False, repr=False)
    _call_count: int = field(default=0, init=False, repr=False)
    
    def __post_init__(self):
        self.console = self.console or Console()
        self._client = OpenAI(
            base_url=self.endpoint,
            api_key=self.api_key
        )
        self._logger = PipelineLogger(
            console=self.console,
            verbose=self.verbose,
            collect_stats=True
        )
        # Enable RAG if vector_index is provided
        if self.vector_index is not None:
            self.use_rag = True
        
        # Log initialization
        if self.verbose:
            self._logger.section("Pipeline Initialized")
            with self._logger.indent():
                self._logger.info(f"Endpoint: {self.endpoint}")
                self._logger.info(f"Model: {self.model}")
                self._logger.info(f"Temperature: {self.temperature}")
                self._logger.info(f"Max Tokens: {self.max_tokens}")
                self._logger.info(f"Samples: {self.num_samples}")
                self._logger.info(f"RAG Enabled: {self.use_rag}")
    
    def generate(
        self,
        system_prompt: str,
        personas: Dict[str, Any] = None,
        guardrails: List[Dict] = None,
        tools: List[Dict] = None,
        scenarios: List[str] = None,
        output_path: Path = None
    ) -> List[Dict]:
        """Generate synthetic training data."""
        personas = personas or {}
        guardrails = guardrails or []
        tools = tools or []
        
        # Default scenarios if not provided
        if not scenarios:
            scenarios = self._generate_default_scenarios(tools)
        
        if self.verbose:
            self._logger.section("Starting Synthetic Data Generation")
            with self._logger.indent():
                self._logger.info(f"Scenarios: {len(scenarios)}")
                self._logger.info(f"Target samples: {self.num_samples}")
        
        if LANGGRAPH_AVAILABLE:
            result = self._run_langgraph_pipeline(
                system_prompt, personas, guardrails, tools, scenarios, output_path
            )
        else:
            result = self._run_simple_pipeline(
                system_prompt, personas, guardrails, tools, scenarios, output_path
            )
        
        # Print final stats
        if self.verbose:
            self._logger.print_stats_summary()
        
        return result
    
    def _generate_default_scenarios(self, tools: List[Dict]) -> List[str]:
        """Generate default scenarios based on available tools."""
        scenarios = [
            "A customer asking a general question",
            "A customer with a complaint",
            "A customer requesting help with their account",
            "A confused customer needing guidance",
            "A customer expressing frustration",
        ]
        
        # Add tool-specific scenarios
        for tool in tools:
            tool_name = tool.get("name", "")
            if "order" in tool_name.lower():
                scenarios.extend([
                    "A customer asking about their order status",
                    "A customer who can't find their order",
                ])
            elif "refund" in tool_name.lower():
                scenarios.extend([
                    "A customer requesting a refund",
                    "A customer unhappy with a product wanting money back",
                ])
            elif "ticket" in tool_name.lower():
                scenarios.extend([
                    "A customer with a complex issue requiring escalation",
                ])
        
        return scenarios[:self.num_samples]
    
    def _run_langgraph_pipeline(
        self,
        system_prompt: str,
        personas: Dict[str, Any],
        guardrails: List[Dict],
        tools: List[Dict],
        scenarios: List[str],
        output_path: Path
    ) -> List[Dict]:
        """Run the full LangGraph pipeline with enhanced nodes."""
        
        # Build the graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes - Enhanced pipeline
        workflow.add_node("expand_scenarios", self._node_expand_scenarios)
        workflow.add_node("generate_conversations", self._node_generate_conversations)
        workflow.add_node("generate_multi_turn", self._node_generate_multi_turn)
        workflow.add_node("score_quality", self._node_score_quality)
        workflow.add_node("validate_guardrails", self._node_validate_guardrails)
        workflow.add_node("filter_and_dedupe", self._node_filter_and_dedupe)
        workflow.add_node("format_dataset", self._node_format_dataset)
        
        # Define edges - Linear flow with all quality stages
        workflow.set_entry_point("expand_scenarios")
        workflow.add_edge("expand_scenarios", "generate_conversations")
        workflow.add_edge("generate_conversations", "generate_multi_turn")
        workflow.add_edge("generate_multi_turn", "score_quality")
        workflow.add_edge("score_quality", "validate_guardrails")
        workflow.add_edge("validate_guardrails", "filter_and_dedupe")
        workflow.add_edge("filter_and_dedupe", "format_dataset")
        workflow.add_edge("format_dataset", END)
        
        # Compile
        app = workflow.compile()
        
        # Initial state
        initial_state: PipelineState = {
            "system_prompt": system_prompt,
            "personas": personas,
            "guardrails": guardrails,
            "tools": tools,
            "config": {
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "num_samples": self.num_samples,
                "multi_turn_ratio": self.multi_turn_ratio,
            },
            "scenarios": scenarios,
            "expanded_scenarios": [],
            "conversations": [],
            "multi_turn_conversations": [],
            "scored_conversations": [],
            "filtered_conversations": [],
            "dataset": [],
            "stats": {},
            "errors": [],
        }
        
        # Run pipeline
        if self.verbose:
            self.console.print("[dim]Running LangGraph pipeline...[/]")
        
        final_state = app.invoke(initial_state)
        
        # Write output
        if output_path:
            self._write_jsonl(final_state["dataset"], output_path)
        
        # Print stats
        if self.verbose:
            stats = final_state.get("stats", {})
            self.console.print(f"  [dim]Stats: {json.dumps(stats, indent=2)}[/]")
        
        return final_state["dataset"]
    
    def _run_simple_pipeline(
        self,
        system_prompt: str,
        personas: Dict[str, Any],
        guardrails: List[Dict],
        tools: List[Dict],
        scenarios: List[str],
        output_path: Path
    ) -> List[Dict]:
        """Run a simple sequential pipeline (fallback when LangGraph not available)."""
        
        if self.verbose:
            self.console.print("[dim]Running simple pipeline (LangGraph not installed)...[/]")
        
        dataset = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Generating conversations...",
                total=len(scenarios)
            )
            
            for i, scenario in enumerate(scenarios):
                try:
                    conversation = self._generate_single_conversation(
                        system_prompt, scenario, personas, tools
                    )
                    
                    if conversation and self._passes_quality_check(conversation):
                        dataset.append(self._format_for_training(
                            system_prompt, conversation
                        ))
                    
                    progress.update(task, advance=1)
                    
                except (APIError, APIConnectionError, RateLimitError) as e:
                    if self.verbose:
                        self.console.print(f"Warning: Failed scenario {i+1}: {e}", style="yellow")
        
        # Write output
        if output_path:
            self._write_jsonl(dataset, output_path)
        
        return dataset
    
    # ========== LANGGRAPH NODES ==========
    
    def _node_expand_scenarios(self, state: PipelineState) -> PipelineState:
        """Node 1: Expand base scenarios with more context and variations.
        
        Creates enough variations to meet the target num_samples by generating
        multiple emotion/complexity combinations per base scenario.
        """
        expanded = []
        
        # Customer emotions/tones to mix in
        emotions = ["neutral", "frustrated", "confused", "happy", "urgent", "polite"]
        
        # Complexity levels
        complexities = ["simple", "moderate", "complex"]
        
        # Calculate how many variations per scenario to reach num_samples
        num_scenarios = len(state["scenarios"])
        target_samples = state["config"].get("num_samples", 10)
        
        # Generate at least 1 variation per scenario, more to reach target
        variations_per_scenario = max(1, (target_samples + num_scenarios - 1) // num_scenarios)
        
        for scenario in state["scenarios"]:
            # Track used combinations to avoid duplicates within same scenario
            used_combinations = set()
            
            for _ in range(variations_per_scenario):
                # Keep trying until we get a unique combination (or give up)
                for attempt in range(10):
                    emotion = random.choice(emotions)
                    complexity = random.choice(complexities)
                    combo = (emotion, complexity)
                    
                    if combo not in used_combinations or attempt == 9:
                        used_combinations.add(combo)
                        expanded.append({
                            "base_scenario": scenario,
                            "emotion": emotion,
                            "complexity": complexity,
                            "context": f"{scenario} (customer is {emotion}, {complexity} issue)"
                        })
                        break
        
        # Shuffle to mix scenarios, then trim to exact target
        random.shuffle(expanded)
        expanded = expanded[:target_samples]
        
        state["expanded_scenarios"] = expanded
        state["stats"]["scenarios_expanded"] = len(expanded)
        return state
    
    def _node_generate_conversations(self, state: PipelineState) -> PipelineState:
        """Node 2: Generate single-turn conversations from expanded scenarios."""
        conversations = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=self.console
        ) as progress:
            
            task = progress.add_task(
                "[cyan]Generating conversations...",
                total=len(state["expanded_scenarios"])
            )
            
            for scenario_data in state["expanded_scenarios"]:
                try:
                    conv = self._generate_single_conversation(
                        state["system_prompt"],
                        scenario_data["context"],
                        state["personas"],
                        state["tools"],
                        emotion=scenario_data.get("emotion", "neutral")
                    )
                    if conv:
                        conv["scenario_data"] = scenario_data
                        conversations.append(conv)
                except (APIError, APIConnectionError, RateLimitError) as e:
                    state["errors"].append(f"Generation error: {e}")
                
                progress.update(task, advance=1)
        
        state["conversations"] = conversations
        state["stats"]["single_turn_generated"] = len(conversations)
        return state
    
    def _node_generate_multi_turn(self, state: PipelineState) -> PipelineState:
        """Node 3: Generate multi-turn conversations for a subset."""
        multi_turn = []
        
        # Select subset for multi-turn
        num_multi = int(len(state["conversations"]) * state["config"]["multi_turn_ratio"])
        selected = random.sample(
            state["conversations"], 
            min(num_multi, len(state["conversations"]))
        )
        
        if self.verbose and selected:
            self.console.print(f"  [dim]Generating {len(selected)} multi-turn conversations...[/]")
        
        for conv in selected:
            try:
                multi_conv = self._extend_to_multi_turn(
                    state["system_prompt"],
                    conv
                )
                if multi_conv:
                    multi_turn.append(multi_conv)
            except (APIError, APIConnectionError, RateLimitError) as e:
                state["errors"].append(f"Multi-turn error: {e}")
        
        state["multi_turn_conversations"] = multi_turn
        state["stats"]["multi_turn_generated"] = len(multi_turn)
        
        # Combine single and multi-turn
        all_convs = state["conversations"] + multi_turn
        state["conversations"] = all_convs
        return state
    
    def _node_score_quality(self, state: PipelineState) -> PipelineState:
        """Node 4: Score conversation quality using heuristics."""
        scored = []
        
        for conv in state["conversations"]:
            score = self._calculate_quality_score(conv, state["system_prompt"])
            conv["quality_score"] = score
            scored.append(conv)
        
        # Sort by quality score
        scored.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
        
        state["scored_conversations"] = scored
        state["stats"]["avg_quality_score"] = (
            sum(c.get("quality_score", 0) for c in scored) / len(scored)
            if scored else 0
        )
        return state
    
    def _node_validate_guardrails(self, state: PipelineState) -> PipelineState:
        """Node 5: Validate conversations against guardrails."""
        validated = []
        violations = 0
        
        for conv in state["scored_conversations"]:
            is_valid, violation_reason = self._check_guardrails(
                conv, 
                state["guardrails"]
            )
            
            if is_valid:
                validated.append(conv)
            else:
                violations += 1
                if self.verbose:
                    self.console.print(f"  [yellow]Guardrail violation:[/] {violation_reason}")
        
        state["scored_conversations"] = validated
        state["stats"]["guardrail_violations"] = violations
        return state
    
    def _node_filter_and_dedupe(self, state: PipelineState) -> PipelineState:
        """Node 6: Filter low quality and deduplicate."""
        filtered = []
        seen_hashes = set()
        
        # Quality threshold (lowered since type-specific scoring is more accurate)
        min_score = 0.4
        
        for conv in state["scored_conversations"]:
            # Quality filter
            if conv.get("quality_score", 0) < min_score:
                continue
            
            # Basic quality check
            if not self._passes_quality_check(conv):
                continue
            
            # Deduplication by content hash (SHA-256, not that MD5 garbage)
            content_hash = hashlib.sha256(
                (conv.get("user", "") + conv.get("assistant", "")).encode()
            ).hexdigest()
            
            if content_hash in seen_hashes:
                continue
            
            seen_hashes.add(content_hash)
            filtered.append(conv)
        
        state["filtered_conversations"] = filtered
        state["stats"]["filtered_count"] = len(filtered)
        state["stats"]["removed_count"] = len(state["scored_conversations"]) - len(filtered)
        return state
    
    def _node_format_dataset(self, state: PipelineState) -> PipelineState:
        """Node 7: Format conversations for training."""
        dataset = []
        
        for conv in state["filtered_conversations"]:
            entry = self._format_for_training(state["system_prompt"], conv)
            dataset.append(entry)
        
        state["dataset"] = dataset
        state["stats"]["final_count"] = len(dataset)
        return state
    
    # ========== HELPER METHODS ==========
    
    def _strip_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> tags from model output.
        
        Handles both complete and unclosed thinking blocks using regex
        like a civilized developer would.
        """
        if '<think>' not in text:
            return text
        
        # Remove complete <think>...</think> blocks (non-greedy, handles nested)
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        
        # Remove unclosed <think> tags (model got cut off mid-thought)
        text = re.sub(r'<think>.*$', '', text, flags=re.DOTALL)
        
        return text.strip()
    
    def _generate_single_conversation(
        self,
        system_prompt: str,
        scenario: str,
        personas: Dict[str, Any],
        tools: List[Dict],
        emotion: str = "neutral"
    ) -> Optional[Dict]:
        """Generate a single conversation for a scenario."""
        
        # Detect agent type for appropriate user message generation (cached, prints once)
        if self._agent_type is None:
            self._agent_type = self._detect_agent_type(system_prompt)
            if self.verbose:
                # Print agent type detection result ONCE
                self.console.print(f"  [dim]Detected agent type: {self._agent_type}[/]")
        
        # Generate type-appropriate user prompts
        if self._agent_type == "code_generation":
            user_prompt = f"""Generate a realistic developer request for this task:
Task: {scenario}

Write ONLY the developer's request, as if you are a developer asking for help.
Be specific about what you need. Include relevant technical context.
Do not include any labels or prefixes - just write the request directly.

Example good requests:
- "I need to write a test for the login page that checks email validation"
- "Can you help me create a function that handles API pagination?"
- "How do I mock the authentication API in my Playwright tests?"

Now write a request for: {scenario}"""
        
        elif self._agent_type == "educational":
            user_prompt = f"""Generate a realistic student question for this topic:
Topic: {scenario}

Write ONLY the student's question, as if you are a student asking to learn.
Be curious and specific about what you want to understand.
Do not include any labels or prefixes - just write the question directly."""
        
        elif self._agent_type == "data_analysis":
            user_prompt = f"""Generate a realistic analyst request for this task:
Task: {scenario}

Write ONLY the analyst's request, as if you are requesting data analysis help.
Be specific about the data, metrics, or insights you need.
Do not include any labels or prefixes - just write the request directly."""
        
        else:  # customer_support or general
            emotion_hints = {
                "frustrated": "The user is clearly frustrated and wants quick resolution.",
                "confused": "The user is confused and needs clear explanations.",
                "happy": "The user is in a good mood and appreciative.",
                "urgent": "The user has an urgent issue that needs immediate attention.",
                "polite": "The user is very polite and patient.",
                "neutral": "The user has a neutral, matter-of-fact tone."
            }
            emotion_context = emotion_hints.get(emotion, "")
            
            user_prompt = f"""Generate a realistic user message for this scenario:
Scenario: {scenario}
{emotion_context}

Write ONLY the user's message, as if you are the user. Be natural and conversational.
Do not include any labels or prefixes like "User:" - just write the message directly."""

        try:
            self._call_count += 1
            
            # Step 1: Generate user message
            user_system_content = "You are simulating a customer. Write realistic customer messages. Output ONLY the message, no thinking or explanations."
            input_chars = len(user_system_content) + len(user_prompt)
            
            start_time = self._logger.log_api_call_start(
                endpoint=self.endpoint,
                model=self.model,
                operation=f"chat/user-gen #{self._call_count}",
                input_chars=input_chars
            )
            
            user_response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": user_system_content},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
                max_tokens=200
            )
            
            # Extract token usage if available
            usage = user_response.usage
            request_tokens = usage.prompt_tokens if usage else input_chars // 4
            response_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else request_tokens + response_tokens
            
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation=f"chat/user-gen #{self._call_count}",
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                success=True
            )
            
            user_message_raw = user_response.choices[0].message.content.strip()
            user_message = self._strip_thinking_tags(user_message_raw)
            
            if self.verbose and user_message != user_message_raw:
                self._logger.info("Stripped thinking tags from user message")
            
            # Fallback if stripping left empty message
            if not user_message and user_message_raw:
                if "</think>" in user_message_raw:
                    user_message = user_message_raw.split("</think>")[-1].strip()
                if not user_message:
                    # Type-appropriate fallback
                    if self._agent_type == "code_generation":
                        user_message = f"Can you help me {scenario.lower()}? I need a working example with best practices."
                    elif self._agent_type == "educational":
                        user_message = f"Can you explain {scenario.lower()}? I want to understand how it works."
                    elif self._agent_type == "data_analysis":
                        user_message = f"I need help with {scenario.lower()}. What insights can you provide?"
                    else:
                        user_message = f"Hi, I need help with {scenario.lower()}."
            
            # Step 2: Generate assistant response (with optional RAG context)
            # Build system prompt with RAG context if available
            effective_system_prompt = system_prompt
            if self.use_rag:
                rag_start = time.time()
                rag_context = self._retrieve_rag_context(user_message)
                rag_duration = (time.time() - rag_start) * 1000
                if rag_context:
                    effective_system_prompt = f"""{system_prompt}

---
RELEVANT CONTEXT (use this to inform your response):
{rag_context}
---"""
                    self._logger.log_rag_retrieval(
                        query_preview=user_message,
                        results_count=rag_context.count("[Context"),
                        top_score=0.0,  # Score logged in retrieval method
                        duration_ms=rag_duration
                    )
            
            self._call_count += 1
            assistant_input_chars = len(effective_system_prompt) + len(user_message)
            
            start_time = self._logger.log_api_call_start(
                endpoint=self.endpoint,
                model=self.model,
                operation=f"chat/assistant-gen #{self._call_count}",
                input_chars=assistant_input_chars
            )
            
            assistant_response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=self.max_tokens
            )
            
            # Extract token usage
            usage = assistant_response.usage
            request_tokens = usage.prompt_tokens if usage else assistant_input_chars // 4
            response_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else request_tokens + response_tokens
            
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation=f"chat/assistant-gen #{self._call_count}",
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                total_tokens=total_tokens,
                success=True
            )
            
            assistant_message_raw = assistant_response.choices[0].message.content.strip()
            assistant_message = self._strip_thinking_tags(assistant_message_raw)
            
            if self.verbose and assistant_message != assistant_message_raw:
                self._logger.info("Stripped thinking tags from assistant message")
            
            if not assistant_message:
                self._logger.warning("Empty response after stripping, skipping")
                return None
            
            return {
                "scenario": scenario,
                "emotion": emotion,
                "user": user_message,
                "assistant": assistant_message,
                "turn_count": 1,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except AuthenticationError as e:
            self._logger.log_connectivity_error(
                endpoint=self.endpoint,
                error_type="AuthenticationError",
                message="Invalid API key for LM Studio"
            )
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation="chat",
                success=False,
                error="Authentication failed"
            )
            return None
        except APIConnectionError as e:
            self._logger.log_connectivity_error(
                endpoint=self.endpoint,
                error_type="APIConnectionError",
                message=f"Cannot connect to LM Studio - is it running? ({str(e)[:100]})"
            )
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation="chat",
                success=False,
                error="Connection failed"
            )
            return None
        except RateLimitError as e:
            self._logger.log_connectivity_error(
                endpoint=self.endpoint,
                error_type="RateLimitError",
                message="Rate limited by API"
            )
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation="chat",
                success=False,
                error="Rate limited"
            )
            return None
        except APIError as e:
            self._logger.log_connectivity_error(
                endpoint=self.endpoint,
                error_type="APIError",
                message=str(e.message) if hasattr(e, 'message') else str(e)
            )
            self._logger.log_api_call_end(
                start_time=start_time,
                endpoint=self.endpoint,
                model=self.model,
                operation="chat",
                success=False,
                error=str(e)
            )
            return None
    
    def _extend_to_multi_turn(
        self,
        system_prompt: str,
        base_conv: Dict
    ) -> Optional[Dict]:
        """Extend a single-turn conversation to multi-turn."""
        
        try:
            # Generate follow-up user message
            follow_up_prompt = f"""The customer just received this response:
"{base_conv['assistant']}"

Generate a realistic follow-up message from the customer. They might:
- Ask for clarification
- Provide additional information requested
- Express satisfaction or continued concern
- Ask a related question

Write ONLY the customer's follow-up message."""

            follow_up_response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are simulating a customer in an ongoing support conversation."},
                    {"role": "user", "content": follow_up_prompt}
                ],
                temperature=self.temperature,
                max_tokens=150
            )
            follow_up_raw = follow_up_response.choices[0].message.content.strip()
            follow_up_user = self._strip_thinking_tags(follow_up_raw)
            
            if not follow_up_user:
                return None
            
            # Generate second assistant response
            second_response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": base_conv["user"]},
                    {"role": "assistant", "content": base_conv["assistant"]},
                    {"role": "user", "content": follow_up_user}
                ],
                temperature=0.7,
                max_tokens=self.max_tokens
            )
            second_assistant_raw = second_response.choices[0].message.content.strip()
            second_assistant = self._strip_thinking_tags(second_assistant_raw)
            
            if not second_assistant:
                return None
            
            return {
                "scenario": base_conv.get("scenario", ""),
                "emotion": base_conv.get("emotion", "neutral"),
                "messages": [
                    {"role": "user", "content": base_conv["user"]},
                    {"role": "assistant", "content": base_conv["assistant"]},
                    {"role": "user", "content": follow_up_user},
                    {"role": "assistant", "content": second_assistant}
                ],
                "turn_count": 2,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except (APIError, APIConnectionError, RateLimitError) as e:
            if self.verbose:
                self.console.print(f"Multi-turn error: {e}", style="yellow")
            return None
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding from local model (cached)."""
        # Create cache key from content hash
        cache_key = hashlib.sha256(text.encode()).hexdigest()[:16]
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        try:
            response = self._client.embeddings.create(
                model=self.embedding_model,
                input=text[:8000]  # Truncate to avoid token limits
            )
            embedding = response.data[0].embedding
            self._embedding_cache[cache_key] = embedding
            return embedding
        except AuthenticationError:
            if self.verbose:
                self.console.print("[red]Embedding error:[/] Invalid API key")
            return None
        except APIConnectionError:
            if self.verbose:
                self.console.print("[yellow]Embedding error:[/] Cannot connect to LM Studio")
            return None
        except RateLimitError:
            if self.verbose:
                self.console.print("[yellow]Embedding error:[/] Rate limited")
            return None
        except APIError as e:
            if self.verbose:
                self.console.print(f"[yellow]Embedding API error:[/] {e.message}")
            return None
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0
    
    def _retrieve_rag_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Retrieve relevant context using RAG from the vector index.
        
        Args:
            query: The query text (scenario or user message)
            top_k: Number of chunks to retrieve (default: self.rag_top_k)
            
        Returns:
            Formatted context string from retrieved chunks
        """
        if not self.use_rag or self.vector_index is None:
            return ""
        
        top_k = top_k or self.rag_top_k
        
        try:
            # Get query embedding
            query_embedding = self._get_embedding(query)
            if query_embedding is None:
                return ""
            
            # Search the index
            results = self.vector_index.search(
                query_embedding=query_embedding,
                top_k=top_k,
                min_score=0.3  # Minimum similarity threshold
            )
            
            if not results:
                return ""
            
            # Format retrieved context
            context_parts = []
            for i, result in enumerate(results, 1):
                if result.chunk is not None:
                    source = result.chunk.source_file
                    content = result.chunk.content[:500]  # Truncate long chunks
                    context_parts.append(
                        f"[Context {i} from {source} (relevance: {result.score:.2f})]:\n{content}"
                    )
            
            if context_parts:
                return "\n\n---\n\n".join(context_parts)
            
            return ""
            
        except (ValueError, KeyError, TypeError) as e:
            if self.verbose:
                self.console.print(f"RAG retrieval error (data): {e}", style="yellow")
            return ""
        except (APIError, APIConnectionError) as e:
            if self.verbose:
                self.console.print(f"RAG retrieval error (API): {e}", style="yellow")
            return ""
    
    def _detect_agent_type_embeddings(self, system_prompt: str) -> Optional[str]:
        """Classify agent type using semantic embeddings."""
        # Get embedding for the system prompt
        prompt_embedding = self._get_embedding(system_prompt)
        if prompt_embedding is None:
            return None
        
        # Compute reference embeddings if not cached
        if not self._ref_embeddings:
            for agent_type, description in AGENT_TYPE_DESCRIPTIONS.items():
                ref_emb = self._get_embedding(description)
                if ref_emb:
                    self._ref_embeddings[agent_type] = ref_emb
        
        if not self._ref_embeddings:
            return None
        
        # Find best matching type by cosine similarity
        best_type = "general"
        best_score = -1.0
        
        for agent_type, ref_embedding in self._ref_embeddings.items():
            similarity = self._cosine_similarity(prompt_embedding, ref_embedding)
            if similarity > best_score:
                best_score = similarity
                best_type = agent_type
        
        return best_type
    
    def _detect_agent_type_keywords(self, system_prompt: str) -> str:
        """Fallback: Detect agent type using keyword matching (legacy)."""
        prompt_lower = system_prompt.lower()
        
        # Code generation indicators
        code_indicators = [
            "code", "programming", "developer", "test", "typescript", "javascript",
            "python", "generate", "playwright", "selenium", "api", "function",
            "class", "method", "syntax", "debug", "compile"
        ]
        if sum(1 for ind in code_indicators if ind in prompt_lower) >= 3:
            return "code_generation"
        
        # Customer support indicators
        support_indicators = [
            "customer", "support", "help", "assist", "service", "inquiry",
            "refund", "order", "complaint", "ticket", "escalate"
        ]
        if sum(1 for ind in support_indicators if ind in prompt_lower) >= 3:
            return "customer_support"
        
        # Educational/tutoring indicators
        education_indicators = [
            "teach", "learn", "explain", "student", "lesson", "tutorial",
            "education", "concept", "understand", "example"
        ]
        if sum(1 for ind in education_indicators if ind in prompt_lower) >= 3:
            return "educational"
        
        # Data/analysis indicators
        data_indicators = [
            "data", "analysis", "report", "chart", "statistics", "metrics",
            "insight", "dashboard", "query", "database"
        ]
        if sum(1 for ind in data_indicators if ind in prompt_lower) >= 3:
            return "data_analysis"
        
        return "general"
    
    def _detect_agent_type(self, system_prompt: str) -> str:
        """Detect agent type using embeddings with keyword fallback."""
        if self.use_embeddings:
            result = self._detect_agent_type_embeddings(system_prompt)
            if result:
                return result
            if self.verbose:
                self.console.print("  [yellow]Embeddings unavailable, using keyword fallback[/]")
        
        return self._detect_agent_type_keywords(system_prompt)
    
    def _calculate_quality_score(self, conv: Dict, system_prompt: str) -> float:
        """Calculate a quality score for a conversation (0-1).
        
        Adapts scoring based on detected agent type.
        """
        # Get messages
        if "messages" in conv:  # Multi-turn
            user_msgs = [m["content"] for m in conv["messages"] if m["role"] == "user"]
            asst_msgs = [m["content"] for m in conv["messages"] if m["role"] == "assistant"]
            user = " ".join(user_msgs)
            assistant = " ".join(asst_msgs)
        else:  # Single-turn
            user = conv.get("user", "")
            assistant = conv.get("assistant", "")
        
        # Use cached agent type, or detect if not yet cached
        agent_type = self._agent_type or self._detect_agent_type(system_prompt)
        
        # Route to type-specific scoring
        if agent_type == "code_generation":
            return self._score_code_generation(user, assistant, conv)
        elif agent_type == "customer_support":
            return self._score_customer_support(user, assistant, conv)
        elif agent_type == "educational":
            return self._score_educational(user, assistant, conv)
        else:
            return self._score_general(user, assistant, conv)
    
    def _score_code_generation(self, user: str, assistant: str, conv: Dict) -> float:
        """Quality scoring for code generation agents."""
        score = 0.0
        asst_len = len(assistant)
        
        # Has code blocks (critical for code generation)
        if "```" in assistant:
            score += 0.35
            # Bonus for language-specific code blocks
            if any(lang in assistant for lang in ["```typescript", "```javascript", "```python", "```java"]):
                score += 0.1
        
        # Length score (code responses tend to be longer)
        if 200 <= asst_len <= 2000:
            score += 0.2
        elif 100 <= asst_len <= 3000:
            score += 0.1
        
        # Has explanation/comments
        explanation_markers = ["//", "/*", "#", "note:", "explanation:", "here's", "this"]
        if any(marker in assistant.lower() for marker in explanation_markers):
            score += 0.1
        
        # Has structure (headers, bullets for explanation)
        if any(marker in assistant for marker in ["##", "**", "- ", "1.", "###"]):
            score += 0.1
        
        # Penalize very short (likely incomplete)
        if asst_len < 100:
            score -= 0.3
        
        # Penalize no code at all
        if "```" not in assistant and not any(kw in assistant for kw in ["function", "const ", "let ", "var ", "class ", "def ", "import "]):
            score -= 0.2
        
        # Multi-turn bonus
        if conv.get("turn_count", 1) > 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_customer_support(self, user: str, assistant: str, conv: Dict) -> float:
        """Quality scoring for customer support agents."""
        score = 0.0
        asst_len = len(assistant)
        
        # Length score (prefer medium-length responses)
        if 100 <= asst_len <= 800:
            score += 0.25
        elif 50 <= asst_len <= 1200:
            score += 0.15
        elif asst_len > 0:
            score += 0.05
        
        # Greeting/acknowledgment
        greetings = ["hi", "hello", "thank", "glad", "happy to help", "understand"]
        if any(g in assistant.lower() for g in greetings):
            score += 0.15
        
        # Has clear structure
        if any(marker in assistant for marker in ["- ", "• ", "1.", "**"]):
            score += 0.15
        
        # Asks clarifying questions
        if "?" in assistant:
            score += 0.1
        
        # Offers further help
        if any(phrase in assistant.lower() for phrase in [
            "anything else", "let me know", "here to help", "further assistance"
        ]):
            score += 0.1
        
        # Empathy markers
        if any(phrase in assistant.lower() for phrase in [
            "understand", "sorry", "appreciate", "frustrat"
        ]):
            score += 0.1
        
        # Penalize very short or very long
        if asst_len < 30:
            score -= 0.3
        if asst_len > 2000:
            score -= 0.1
        
        # Multi-turn bonus
        if conv.get("turn_count", 1) > 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_educational(self, user: str, assistant: str, conv: Dict) -> float:
        """Quality scoring for educational/tutoring agents."""
        score = 0.0
        asst_len = len(assistant)
        
        # Length (educational content tends to be detailed)
        if 200 <= asst_len <= 1500:
            score += 0.2
        elif 100 <= asst_len <= 2000:
            score += 0.1
        
        # Has examples
        if any(marker in assistant.lower() for marker in ["example", "for instance", "such as", "e.g."]):
            score += 0.2
        
        # Has structure (good for learning)
        if any(marker in assistant for marker in ["##", "**", "- ", "1.", "###"]):
            score += 0.15
        
        # Explanatory language
        if any(phrase in assistant.lower() for phrase in [
            "this means", "because", "therefore", "in other words", "simply put"
        ]):
            score += 0.15
        
        # Code examples for technical topics
        if "```" in assistant:
            score += 0.15
        
        # Questions to check understanding
        if "?" in assistant:
            score += 0.1
        
        # Penalize very short
        if asst_len < 50:
            score -= 0.3
        
        # Multi-turn bonus
        if conv.get("turn_count", 1) > 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _score_general(self, user: str, assistant: str, conv: Dict) -> float:
        """Generic quality scoring for unclassified agents."""
        score = 0.0
        asst_len = len(assistant)
        
        # Length score
        if 100 <= asst_len <= 1500:
            score += 0.25
        elif 50 <= asst_len <= 2000:
            score += 0.15
        elif asst_len > 0:
            score += 0.05
        
        # Has structure
        if any(marker in assistant for marker in ["- ", "• ", "1.", "**", "##"]):
            score += 0.2
        
        # Code blocks (if present, likely intentional)
        if "```" in assistant:
            score += 0.15
        
        # Responsive to user
        if "?" in assistant or any(word in assistant.lower() for word in user.lower().split()[:3]):
            score += 0.15
        
        # Professional tone markers
        if any(phrase in assistant.lower() for phrase in [
            "here", "following", "below", "this", "let me"
        ]):
            score += 0.1
        
        # Penalize very short
        if asst_len < 30:
            score -= 0.3
        
        # Penalize repetition
        words = assistant.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.4:
                score -= 0.2
        
        # Multi-turn bonus
        if conv.get("turn_count", 1) > 1:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _check_guardrails(
        self,
        conv: Dict,
        guardrails: List[Dict]
    ) -> tuple[bool, Optional[str]]:
        """Check if conversation violates any guardrails."""
        
        # Get assistant response(s)
        if "messages" in conv:
            assistant_text = " ".join(
                m["content"] for m in conv["messages"] if m["role"] == "assistant"
            )
        else:
            assistant_text = conv.get("assistant", "")
        
        assistant_lower = assistant_text.lower()
        
        # Basic safety checks
        unsafe_patterns = [
            "i cannot help with illegal",
            "i won't assist with",
            "this is inappropriate",
            "i'm not able to provide",
            "violates our policies",
        ]
        
        # These are actually GOOD - the model refused unsafe requests
        # But too many refusals might indicate bad user messages
        refusal_count = sum(1 for p in unsafe_patterns if p in assistant_lower)
        if refusal_count > 2:
            return False, "Too many refusals (possible bad user message)"
        
        # Check for PII leakage patterns (exclude company emails from system prompt)
        pii_patterns = [
            (r'\b\d{3}-\d{2}-\d{4}\b', "SSN pattern"),  # SSN
            (r'\b\d{16}\b', "Credit card pattern"),  # Credit card
        ]
        
        for pattern, name in pii_patterns:
            if re.search(pattern, assistant_text):
                return False, f"Possible PII in response: {name}"
        
        # Check against defined guardrails
        for guardrail in guardrails:
            rules = guardrail.get("rules", {})
            hard_constraints = rules.get("hard_constraints", [])
            
            for constraint in hard_constraints:
                trigger = constraint.get("trigger", "").lower()
                if trigger and trigger in assistant_lower:
                    return False, f"Guardrail violation: {constraint.get('name', 'unknown')}"
        
        return True, None
    
    def _passes_quality_check(self, conversation: Dict) -> bool:
        """Basic quality check for a conversation."""
        
        # Handle multi-turn format
        if "messages" in conversation:
            user_msgs = [m["content"] for m in conversation["messages"] if m["role"] == "user"]
            asst_msgs = [m["content"] for m in conversation["messages"] if m["role"] == "assistant"]
            user = " ".join(user_msgs)
            assistant = " ".join(asst_msgs)
        else:
            assistant = conversation.get("assistant", "")
            user = conversation.get("user", "")
        
        # Minimum length checks
        if len(assistant) < 50:
            return False
        if len(user) < 10:
            return False
        
        # Check for error responses
        error_markers = [
            "I cannot", "I can't", "I'm unable",
            "Error:", "error occurred",
        ]
        
        error_count = sum(1 for marker in error_markers if marker.lower() in assistant.lower())
        if error_count > 2:
            return False
        
        # Check for repetitive content
        if len(assistant) > 100:
            first_100 = assistant[:100]
            if assistant.count(first_100) > 1:
                return False
        
        return True
    
    def _format_for_training(self, system_prompt: str, conversation: Dict) -> Dict:
        """Format a conversation for fine-tuning."""
        
        # Handle multi-turn format
        if "messages" in conversation:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(conversation["messages"])
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": conversation["user"]},
                {"role": "assistant", "content": conversation["assistant"]}
            ]
        
        return {
            "messages": messages,
            "_metadata": {
                "scenario": conversation.get("scenario", ""),
                "emotion": conversation.get("emotion", "neutral"),
                "turn_count": conversation.get("turn_count", 1),
                "quality_score": conversation.get("quality_score", 0),
                "generated_at": conversation.get("generated_at", "")
            }
        }
    
    def _write_jsonl(self, dataset: List[Dict], output_path: Path):
        """Write dataset to JSONL file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
