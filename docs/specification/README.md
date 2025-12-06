# Specification Documents

This directory contains the draft specifications for the COP ecosystem.

## Documents

| Document | Description | Status |
|----------|-------------|--------|
| [package-format.md](./package-format.md) | COP package format specification (`cop.yaml`) | Draft v0.1.0 |
| [architecture.md](./architecture.md) | System architecture diagrams | Draft |

## Package Format Overview

The COP package format defines:

### `cop.yaml` Manifest

```yaml
meta:
  name: "my-context-module"
  version: "1.0.0"
  
compatibility:
  models:
    - name: "gpt-4"
      tested_versions: ["gpt-4-0125-preview"]

context:
  system:
    source: "./prompts/system.md"
  personas:
    default: "friendly"
    available:
      friendly: { source: "./personas/friendly.yaml" }
  guardrails:
    - { source: "./guardrails/safety.yaml", priority: 100 }
  knowledge:
    - { source: "./knowledge/faq.md", type: "static" }
  tools:
    - { source: "./tools/search.yaml" }

dependencies:
  "tone-analyzer": "^2.0.0"

evaluation:
  test_suites:
    - { path: "./tests/behavioral/", type: "llm-judged" }
```

### Directory Structure

```
my-context-module/
├── cop.yaml              # Package manifest
├── cop.lock              # Locked dependencies
├── prompts/              # Prompt templates
├── personas/             # Persona definitions
├── guardrails/           # Behavioral constraints
├── knowledge/            # Static knowledge
├── tools/                # Tool definitions
├── tests/                # Evaluation suites
└── README.md             # Documentation
```

## Architecture Overview

The [architecture document](./architecture.md) contains diagrams for:

1. **High-Level System Architecture** — COP ecosystem overview
2. **Package Resolution Flow** — Dependency resolution algorithm
3. **Build Pipeline** — From source to artifacts
4. **Evaluation Pipeline** — Testing and fingerprinting
5. **CI/CD Integration** — Continuous delivery workflow
6. **Runtime Architecture** — Request processing flow
7. **Registry Architecture** — Package distribution system

## Related

- [Research Documents](../research/) — Background research
- [Build Documents](../build/) — Build process details
- [Example Package](../../examples/customer-support-agent/) — Reference implementation
