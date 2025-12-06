# COP Schema Validation

This directory contains JSON Schema definitions for validating Context-Oriented Programming (COP) configurations.

## Schemas

| Schema | Description | File Types |
|--------|-------------|------------|
| `cop-manifest.schema.json` | Main package manifest | `cop.yaml` |
| `persona.schema.json` | Persona definitions | `personas/*.yaml` |
| `guardrail.schema.json` | Guardrail configurations | `guardrails/*.yaml` |
| `tool.schema.json` | Tool/function definitions | `tools/*.yaml` |
| `test.schema.json` | Test suite configurations | `tests/**/*.yaml` |

## Validation Tool

Use the `validate_cop.py` script to validate COP packages:

```bash
# Install dependencies
pip install pyyaml jsonschema

# Validate a single package
python3 validate_cop.py examples/customer-support-agent

# Validate all packages in examples/
python3 validate_cop.py --all

# Validate a single file against a specific schema
python3 validate_cop.py --file examples/customer-support-agent/personas/professional.yaml --schema persona

# Verbose output with more details
python3 validate_cop.py examples/customer-support-agent --verbose
```

## Schema Overview

### cop-manifest.schema.json

The main manifest schema validates the structure of `cop.yaml` files:

```yaml
meta:
  name: "package-name"          # Required: lowercase with hyphens
  version: "1.0.0"              # Required: semantic version
  description: "..."            # Optional
  author: "..."                 # Optional
  license: "MIT"                # Optional: SPDX identifier

compatibility:
  models: [...]                 # Supported LLM models
  features_required: [...]      # Required features
  context_window: {...}         # Token requirements

context:
  system: {...}                 # System prompt configuration
  personas: {...}               # Available personas
  knowledge: [...]              # Knowledge sources
  guardrails: [...]             # Safety guardrails
  tools: [...]                  # Tool definitions

# ... more sections
```

### persona.schema.json

Persona files define communication styles:

```yaml
name: "persona-name"            # Required
description: "..."              # Optional
tone:
  formality: "low|medium|high"
  warmth: "low|medium|high"
  empathy: "low|medium|high"
communication_style:
  greeting: "informal|formal|casual|minimal|..."
  closing: "informal|formal|warm|minimal|..."
  use_contractions: true|false
  use_emoji: true|false
vocabulary:
  preferred: [...]
  avoid: [...]
response_patterns: {...}
example_exchanges: [...]
```

### guardrail.schema.json

Guardrail files define safety constraints:

```yaml
name: "guardrail-name"          # Required
description: "..."              # Optional
priority: 0-100                 # Required: higher = more important

hard_constraints:
  - name: "constraint-name"
    description: "..."
    rules: [...]
    trigger_phrases: [...]      # Optional

soft_constraints:
  - name: "preference-name"
    behavior: "gentle_redirect|neutral_response|deflect|recommend|..."
    rules: [...]

violation_responses:
  constraint_name:
    response: "Template response"
    escalate: true|false

monitoring:
  log_violations: true|false
  alert_threshold: 3
  escalate_on: [...]
```

### tool.schema.json

Tool files define function calling specs:

```yaml
name: "tool_name"               # Required
description: "..."              # Required
version: "1.0.0"                # Optional

openai_spec:                    # Required: OpenAI function calling format
  type: function
  function:
    name: "function_name"
    description: "..."
    parameters:
      type: object
      properties: {...}
      required: [...]

response_schema: {...}          # Optional
examples: [...]                 # Optional
security: [...]                 # Optional
rate_limits: {...}              # Optional
```

### test.schema.json

Test files define evaluation suites:

```yaml
name: "test-suite-name"         # Required
type: "llm-judged|adversarial|deterministic|comparison"  # Required

judge_model: "gpt-4"            # For llm-judged tests
judge_temperature: 0.0

evaluation_criteria:
  criterion_name:
    weight: 0.0-1.0
    description: "..."
    rubric:
      "5": "Excellent"
      "4": "Good"
      # ...

test_cases:
  - id: "test-id"
    name: "Test name"
    input: {...}
    expected_behaviors: [...]
    minimum_scores: {...}
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All validations passed |
| 1 | Validation errors found |
| 2 | Invalid arguments or missing parameters |

## Error Types

The validator reports several types of issues:

1. **Schema Violations**: Data doesn't match the expected schema
   - Wrong type (e.g., string instead of number)
   - Missing required fields
   - Invalid enum values
   - Pattern mismatches

2. **File Reference Errors**: Referenced files don't exist
   - Missing persona files
   - Missing guardrail files
   - Missing tool files
   - Missing knowledge files

3. **YAML Syntax Errors**: Invalid YAML formatting

## Extending Schemas

The schemas use JSON Schema draft-07 and allow additional properties for extensibility. To add custom fields to your COP files, simply include them in your YAML. The validator will still check required fields and validate known properties.

## Integration

### CI/CD Integration

```yaml
# GitHub Actions example
- name: Validate COP Package
  run: |
    pip install pyyaml jsonschema
    python3 validate_cop.py my-package/
```

### Programmatic Usage

```python
from validate_cop import COPValidator

validator = COPValidator()
result = validator.validate_package(Path("my-package/"))

if result.has_errors:
    for issue in result.issues:
        print(issue.format())
    sys.exit(1)
```
