# Research Documents

This directory contains the core research findings and analysis for Context-Oriented Programming (COP).

## Documents

| Document | Description | Reading Time |
|----------|-------------|--------------|
| [key-findings.md](./key-findings.md) | Distilled insights and executive summary | 5-10 min |
| [main-research.md](./main-research.md) | Comprehensive research report | 30-45 min |
| [deep-analysis.md](./deep-analysis.md) | Philosophical and technical deep-dive | 25-35 min |
| [tool-comparison.md](./tool-comparison.md) | Analysis of LangChain, PromptFlow, Semantic Kernel, etc. | 15-20 min |
| [opinion.md](./opinion.md) | Perspective on COP as a paradigm shift | 10-15 min |

## Quick Start

**New to COP?** Start with [key-findings.md](./key-findings.md) for a quick overview.

**Want comprehensive details?** Read [main-research.md](./main-research.md).

**Interested in the philosophy?** Check [deep-analysis.md](./deep-analysis.md).

## Key Concepts

### What is COP?

Context-Oriented Programming is an emerging paradigm where:
- **Intent** replaces implementation
- **Context modules** (prompts, personas, guardrails, knowledge) are the primary artifacts
- **Behavioral evaluation** replaces deterministic testing
- **LLMs** execute the specified behavior

### The Paradigm Shift

```
Traditional: Code → Compile → Execute → Deterministic Output
COP:         Context → Assemble → Evaluate → Probabilistic Behavior
```

### Critical Gaps Identified

| Gap | Severity |
|-----|----------|
| Standard package format | 🔴 Critical |
| Public registry | 🔴 Critical |
| Dependency management | 🔴 Critical |
| LLM drift handling | 🟠 High |
| Cross-model portability | 🟠 High |

## Related

- [Specification Documents](../specification/) — Package format and architecture
- [Build Documents](../build/) — Understanding the build process
- [Example Package](../../examples/customer-support-agent/) — See COP in action
