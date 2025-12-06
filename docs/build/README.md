# Build Documents

This directory contains deep-dive documentation on what "building" means in Context-Oriented Programming.

## Documents

| Document | Description | Reading Time |
|----------|-------------|--------------|
| [concept.md](./concept.md) | What "build" means in COP — overview and comparison | 25-35 min |
| [internals.md](./internals.md) | Technical implementation details, algorithms, data structures | 45-60 min |

## The Build Process in COP

Unlike traditional compilation, COP builds are about **context assembly, evaluation, and transformation**.

### Traditional Build vs COP Build

| Aspect | Traditional | COP |
|--------|-------------|-----|
| **Input** | Source code | Context modules |
| **Process** | Compile → Link | Assemble → Evaluate → Transform |
| **Output** | Binary/library | Context bundle + provider configs |
| **Determinism** | Fully deterministic | Partially deterministic |
| **Validation** | Type checking | Behavioral evaluation |

### Build Pipeline Stages

```
1. LOAD & PARSE
   └── Parse cop.yaml and load source files

2. DEPENDENCY RESOLUTION
   └── Resolve versions, build dependency graph

3. CONTEXT COMPILATION
   └── Resolve variables, merge prompts, apply personas

4. VALIDATION
   └── Check completeness, conflicts, token limits

5. EVALUATION
   └── Run tests (deterministic, LLM-judged, adversarial)

6. OPTIMIZATION
   └── Minify prompts, compress knowledge, optimize tokens

7. TARGET TRANSFORMATION
   └── Generate OpenAI, Anthropic, Azure formats

8. ARTIFACT GENERATION
   └── Package, sign, checksum
```

### Build Artifacts

A COP build produces:

- **Context Bundle** — Merged, validated context specification
- **Provider Configs** — Target-specific formats (OpenAI, Anthropic, etc.)
- **Evaluation Fingerprint** — Reproducibility metadata
- **Deployment Manifest** — Runtime configuration

## Quick Reference

### Build Determinism

| Component | Deterministic? |
|-----------|----------------|
| Context assembly | ✅ Yes |
| Template compilation | ✅ Yes |
| Dependency resolution | ✅ Yes |
| Evaluation results | ❌ No (probabilistic) |
| Runtime behavior | ⚠️ Varies (model drift) |

### Evaluation Fingerprinting

Enables reproducibility despite probabilistic evaluation:

```json
{
  "package": "customer-support@1.2.0",
  "test_hash": "sha256:abc123...",
  "model_versions": {
    "gpt-4": "gpt-4-0125-preview"
  },
  "scores": {
    "quality": 0.94,
    "safety": 0.99
  }
}
```

## Related

- [Research Documents](../research/) — Background and context
- [Specification Documents](../specification/) — Package format spec
- [Example Package](../../examples/customer-support-agent/) — See it in practice
