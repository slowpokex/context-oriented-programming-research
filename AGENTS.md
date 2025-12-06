# AGENTS.md

> Instructions for AI agents working with this repository.

## Project Summary

This is a **research repository** for Context-Oriented Programming (COP) — a paradigm where LLM applications are built by composing context modules (prompts, personas, guardrails, knowledge, tools) rather than writing procedural code.

**This is NOT a code repository.** It contains:
- Research documentation (Markdown)
- Specification drafts (Markdown)
- Example COP packages (YAML + Markdown)

## Quick Reference

| Path | Purpose |
|------|---------|
| `README.md` | Project overview |
| `RESEARCH-FINDINGS.md` | Key insights summary |
| `BUILD-CONCEPT-COP.md` | What "build" means in COP |
| `SPECIFICATION.md` | Draft package format spec |
| `examples/customer-support-agent/` | Reference COP package |

## Repository Structure

```
├── *.md                          # Research documents
├── examples/
│   └── customer-support-agent/   # Example COP package
│       ├── cop.yaml              # Package manifest (start here)
│       ├── prompts/              # System prompts with {{variables}}
│       ├── personas/             # Tone and communication style
│       ├── guardrails/           # Safety constraints
│       ├── knowledge/            # FAQ and static content
│       ├── tools/                # Function definitions
│       └── tests/                # Evaluation test suites
```

## Key Concepts

1. **cop.yaml** — Central manifest defining a COP package (like `package.json` for prompts)
2. **Prompts** — Markdown files with `{{variable}}` template syntax
3. **Personas** — YAML configs defining tone, vocabulary, response patterns
4. **Guardrails** — Safety rules with priorities (higher = more important)
5. **Tools** — OpenAI function-calling compatible definitions
6. **Tests** — Behavioral (LLM-judged) and adversarial (safety) test cases

## File Conventions

### YAML Files
- 2-space indentation
- Include `name`, `description` fields
- Add comments explaining purpose
- Use `priority: 0-100` for guardrails (100 = highest)

### Prompts (Markdown)
- Use `{{variable_name}}` for template variables
- Structure with clear sections: Role, Responsibilities, Guidelines
- Keep escalation triggers explicit

### Tests
- `type: llm-judged` — Quality evaluation with rubrics
- `type: adversarial` — Safety/jailbreak resistance tests
- Include `expected_behaviors` and `failure_indicators`

## Common Tasks

### Add a new persona
1. Create `examples/customer-support-agent/personas/{name}.yaml`
2. Include: `name`, `description`, `tone`, `vocabulary`, `response_patterns`
3. Register in `cop.yaml` under `context.personas.available`

### Add a guardrail
1. Create `examples/customer-support-agent/guardrails/{name}.yaml`
2. Define `hard_constraints` and `soft_constraints`
3. Set appropriate `priority` (safety=100, compliance=90, brand=80)
4. Add `violation_responses` for each constraint
5. Register in `cop.yaml` under `context.guardrails`

### Add a test case
1. Add to appropriate file in `tests/behavioral/` or `tests/safety/`
2. Include `id`, `name`, `input`, `expected_behaviors`
3. For behavioral: add `minimum_scores`
4. For safety: add `failure_indicators`

### Add documentation
1. Follow existing Markdown structure
2. Use tables for comparisons
3. Include practical examples
4. Update `README.md` if adding new top-level documents

## Guidelines

### Do
- Keep prompts concise (token efficiency matters)
- Test safety implications of changes
- Use existing patterns from the example package
- Update related documentation when making changes

### Don't
- Add code implementations (this is research/spec only)
- Create deeply nested structures in YAML
- Remove safety guardrails without justification
- Use provider-specific features in core definitions

## Understanding COP Build Process

Unlike traditional compilation, COP "build" means:

```
Context Modules → Assembly → Evaluation → Transformation → Artifacts
```

1. Parse `cop.yaml` and load all referenced files
2. Resolve dependencies and merge contexts
3. Validate (variables, conflicts, token limits)
4. Evaluate (run test suites, LLM-as-judge)
5. Transform to target formats (OpenAI, Anthropic, Azure)

## Related Files

- `.cursor/rules/cop-research.mdc` — Cursor IDE rules
- `LICENSE` — Apache 2.0
