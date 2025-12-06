# COP Build System Documentation

This section documents the build process, artifacts, and compilation concepts in Context-Oriented Programming (COP).

## Documents

| Document | Purpose |
|----------|---------|
| [artifacts.md](./artifacts.md) | **Comprehensive analysis of COP build artifacts** — canonical reference for what "build artifacts" are in COP |
| [compilation.md](./compilation.md) | **What "compilation" means in the COP paradigm** — stages, processes, transformations |
| [concept.md](./concept.md) | Overview of the build concept — context assembly, evaluation, transformation |
| [internals.md](./internals.md) | Technical implementation details — data structures, algorithms, performance |

## Reading Order

For a complete understanding of COP build systems:

1. **[artifacts.md](./artifacts.md)** — Start here to understand what build artifacts are (Context Bundle, Fine-tune Dataset, Model Artifact, RAG Index) and which to use
2. **[compilation.md](./compilation.md)** — Understand how "compilation" works in COP vs traditional software
3. **[concept.md](./concept.md)** — Deep-dive into the philosophical and practical aspects of building
4. **[internals.md](./internals.md)** — Technical implementation details for tool developers

## Quick Overview

### What is a COP Build?

Unlike traditional software compilation where source code is transformed into machine-executable binaries, a COP build:

1. **Assembles** context modules (prompts, personas, guardrails, knowledge)
2. **Validates** the assembled context for completeness and conflicts
3. **Evaluates** behavioral quality through LLM-judged tests
4. **Transforms** the context into provider-specific formats
5. **Produces** deployable artifacts with reproducibility metadata

### Build Artifacts

COP supports multiple artifact types for different use cases:

| Artifact Type | Primary Use Case | Portability |
|---------------|------------------|-------------|
| **Context Bundle** | Development, CI/CD | High (provider-agnostic) |
| **Fine-tune Dataset** | Model customization | Medium (model-specific) |
| **Model Artifact** | Production optimization | Low (frozen weights) |
| **RAG Index** | Knowledge retrieval | Medium (embedding-model-specific) |

### Recommended Default

**The Context Bundle is the recommended canonical COP build artifact** for most use cases. See [artifacts.md](./artifacts.md) for detailed analysis and the comparison table.

### The `cop build` Command

```bash
# Basic build (context bundle only)
cop build

# Build with evaluation
cop build --evaluate

# Build for specific target
cop build --target openai

# Build all secondary artifacts
cop build --artifact all
```

See [artifacts.md](./artifacts.md) Section 7 for complete pipeline documentation.

## Related Documentation

- [Package Format Specification](../specification/package-format.md)
- [Architecture Diagrams](../specification/architecture.md)
- [Deep Analysis](../research/deep-analysis.md)
- [Tool Comparison](../research/tool-comparison.md)
