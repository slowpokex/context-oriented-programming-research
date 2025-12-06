# COP Package Format Specification

## Draft Version 0.1.0

---

## 1. Overview

This document specifies the **COP Package Format** (`.cop`), a standardized format for defining, packaging, and distributing Context-Oriented Programming modules for Large Language Model applications.

### 1.1 Goals

1. **Portability**: Packages should work across different LLM providers
2. **Composability**: Packages should be combinable with clear dependency rules
3. **Testability**: Built-in support for evaluation and quality gates
4. **Versioning**: Track changes with reproducibility guarantees
5. **Simplicity**: Easy to author and understand

### 1.2 Non-Goals

1. Runtime implementation (framework-agnostic)
2. Model training or fine-tuning
3. Data storage beyond context definitions

---

## 2. Package Structure

### 2.1 Directory Layout

```
my-context-module/
├── cop.yaml              # Package manifest (required)
├── cop.lock              # Dependency lock file (generated)
├── prompts/              # Prompt templates
│   ├── system.md
│   └── user.md
├── personas/             # Persona definitions
│   └── default.yaml
├── guardrails/           # Behavioral constraints
│   └── safety.yaml
├── knowledge/            # Static knowledge files
│   └── faq.md
├── tools/                # Tool/function definitions
│   └── search.yaml
├── schemas/              # JSON schemas for validation
│   └── output.schema.json
├── tests/                # Evaluation test suites
│   ├── unit/
│   └── behavioral/
├── .copignore            # Files to exclude from package
└── README.md             # Package documentation
```

### 2.2 File Naming Conventions

| Pattern | Purpose |
|---------|---------|
| `cop.yaml` | Package manifest |
| `cop.lock` | Locked dependencies |
| `*.md` | Markdown prompt templates |
| `*.yaml` | Configuration files |
| `*.schema.json` | JSON Schema definitions |
| `.copignore` | Exclusion patterns |

---

## 3. Manifest Specification (cop.yaml)

### 3.1 Complete Schema

```yaml
# cop.yaml JSON Schema
$schema: "https://cop.dev/schema/cop-manifest-v1.json"

# ─────────────────────────────────────────────────────────
# SECTION 1: Package Metadata
# ─────────────────────────────────────────────────────────

meta:
  # Required fields
  name: string                    # Package name (lowercase, hyphens allowed)
                                  # Pattern: ^[a-z][a-z0-9-]*[a-z0-9]$
                                  # Min length: 2, Max length: 64
  
  version: string                 # Semantic version (MAJOR.MINOR.PATCH)
                                  # Pattern: ^\d+\.\d+\.\d+(-[a-z0-9.]+)?$
  
  # Optional fields
  description: string             # Brief description (max 256 chars)
  author: string | Author         # Author name or object
  license: string                 # SPDX license identifier
  repository: string              # Git repository URL
  homepage: string                # Project homepage
  keywords: string[]              # Search keywords
  
  # Author object format
  # author:
  #   name: string
  #   email: string
  #   url: string

# ─────────────────────────────────────────────────────────
# SECTION 2: LLM Compatibility
# ─────────────────────────────────────────────────────────

compatibility:
  models:
    - name: string                # Model family (gpt-4, claude-3, etc.)
      min_version: string         # Minimum supported version
      tested_versions: string[]   # Actually tested versions
      eval_scores: object         # Evaluation scores per metric
      
  features_required: string[]     # Required LLM features
                                  # Valid: function_calling, json_mode,
                                  # vision, system_messages, streaming
  
  context_window:
    minimum: integer              # Minimum required context (tokens)
    recommended: integer          # Recommended context size

# ─────────────────────────────────────────────────────────
# SECTION 3: Context Definition
# ─────────────────────────────────────────────────────────

context:
  # System prompt configuration
  system:
    source: string                # Path to system prompt file
    variables:                    # Template variables
      <variable_name>:
        type: string              # string, number, boolean, array, object
        required: boolean
        default: any
        description: string
        format: string            # email, url, date, etc. (optional)
        enum: any[]               # Allowed values (optional)
        pattern: string           # Regex pattern (optional)
  
  # Persona configurations
  personas:
    default: string               # Default persona name
    available:
      <persona_name>:
        source: string            # Path to persona file
        description: string
  
  # Knowledge attachments
  knowledge:
    - name: string
      source: string              # Path to knowledge file
      type: string                # static, structured, dynamic
      schema: string              # JSON Schema path (for structured)
      description: string
  
  # Guardrail configurations
  guardrails:
    - name: string
      source: string              # Path to guardrail file
      priority: integer           # Execution priority (higher = first)
      description: string
  
  # Tool definitions
  tools:
    - name: string
      source: string              # Path to tool definition
      description: string
      requires_approval: boolean  # Human approval needed
      rate_limit: integer         # Max calls per session

# ─────────────────────────────────────────────────────────
# SECTION 4: Dependencies
# ─────────────────────────────────────────────────────────

dependencies:
  <package_name>: string          # Version constraint
                                  # Supports: ^1.0.0, ~1.0.0, >=1.0.0,
                                  # 1.0.0 - 2.0.0, 1.0.x

dev_dependencies:
  <package_name>: string          # Development-only dependencies

peer_dependencies:
  <package_name>: string          # Must be provided by consumer

optional_dependencies:
  <package_name>:
    version: string
    condition: string             # When to include

# ─────────────────────────────────────────────────────────
# SECTION 5: Build Configuration
# ─────────────────────────────────────────────────────────

build:
  targets:
    <target_name>:
      format: string              # Output format
      optimize: boolean
      include_knowledge: boolean
      # ... target-specific options
  
  preprocessing:
    - <preprocessor>: boolean | object
  
  postprocessing:
    - <postprocessor>: boolean | object

# ─────────────────────────────────────────────────────────
# SECTION 6: Evaluation Configuration
# ─────────────────────────────────────────────────────────

evaluation:
  framework: string               # Evaluation framework
  
  test_suites:
    - name: string
      path: string
      type: string                # deterministic, llm-judged, adversarial
      judge_model: string         # For llm-judged tests
      description: string
  
  benchmarks:
    - name: string
      metric: string
      threshold: number
      weight: number              # For aggregate scoring
  
  regression:
    baseline: string              # Version to compare against
    tolerance: number             # Acceptable degradation

# ─────────────────────────────────────────────────────────
# SECTION 7: Runtime Configuration
# ─────────────────────────────────────────────────────────

runtime:
  model_config:
    temperature: number           # 0.0 - 2.0
    max_tokens: integer
    top_p: number                 # 0.0 - 1.0
    frequency_penalty: number     # -2.0 - 2.0
    presence_penalty: number      # -2.0 - 2.0
    stop_sequences: string[]
  
  retry_policy:
    max_retries: integer
    backoff: string               # linear, exponential, constant
    initial_delay_ms: integer
    max_delay_ms: integer
  
  fallback:
    enabled: boolean
    trigger: string               # Condition expression
    model: string                 # Fallback model
    notify: boolean
  
  rate_limits:
    requests_per_minute: integer
    tokens_per_minute: integer
  
  cache:
    enabled: boolean
    ttl_seconds: integer
    similarity_threshold: number  # For semantic caching

# ─────────────────────────────────────────────────────────
# SECTION 8: Observability
# ─────────────────────────────────────────────────────────

observability:
  logging:
    level: string                 # debug, info, warn, error
    include_prompts: boolean
    include_responses: boolean
  
  metrics:
    enabled: boolean
    export_format: string         # prometheus, statsd, otlp
  
  tracing:
    enabled: boolean
    sample_rate: number
    export_to: string
```

---

## 4. Version Constraints

### 4.1 Supported Formats

| Format | Meaning | Example |
|--------|---------|---------|
| `1.2.3` | Exact version | Only 1.2.3 |
| `^1.2.3` | Compatible with | 1.2.3 to <2.0.0 |
| `~1.2.3` | Approximately | 1.2.3 to <1.3.0 |
| `>=1.2.3` | Greater or equal | 1.2.3 and above |
| `<2.0.0` | Less than | Below 2.0.0 |
| `1.2.x` | Any patch | 1.2.0 to <1.3.0 |
| `*` | Any version | Latest compatible |
| `1.2.3 - 2.0.0` | Range | 1.2.3 to 2.0.0 |

### 4.2 Version Resolution

```
Algorithm: Dependency Resolution

1. Build dependency graph from all cop.yaml files
2. For each package:
   a. Collect all version constraints
   b. Find versions satisfying ALL constraints
   c. Select highest satisfying version
3. If conflict exists:
   a. Report conflicting constraints
   b. Suggest resolution strategies
4. Generate cop.lock with resolved versions
```

---

## 5. Lock File Format (cop.lock)

```yaml
# cop.lock - Auto-generated, do not edit

lockfile_version: 1

packages:
  "tone-analyzer@2.1.0":
    version: "2.1.0"
    resolved: "https://registry.cop.dev/tone-analyzer/-/tone-analyzer-2.1.0.cop"
    integrity: "sha512-abc123..."
    dependencies:
      "sentiment-core": "1.5.2"
    
    # LLM evaluation fingerprints
    eval_fingerprints:
      "gpt-4-0125-preview":
        quality_score: 0.92
        eval_date: "2025-01-15"
        eval_hash: "sha256:def456..."
      "claude-3-opus-20240229":
        quality_score: 0.89
        eval_date: "2025-02-01"
        eval_hash: "sha256:ghi789..."

  "sentiment-core@1.5.2":
    version: "1.5.2"
    resolved: "https://registry.cop.dev/sentiment-core/-/sentiment-core-1.5.2.cop"
    integrity: "sha512-xyz789..."
    dependencies: {}
```

---

## 6. Prompt Template Syntax

### 6.1 Variable Interpolation

```markdown
# Simple variable
Hello {{name}}, welcome to {{company_name}}!

# With default value
Contact us at {{support_email|support@example.com}}

# Conditional content
{{#if premium_user}}
You have access to premium features.
{{/if}}

# Loops
Your recent orders:
{{#each orders}}
- {{this.id}}: {{this.status}}
{{/each}}
```

### 6.2 Include Directive

```markdown
# Include another file
{{> ./partials/greeting.md}}

# Include with context
{{> ./partials/product_info.md product=current_product}}
```

### 6.3 Model-Specific Blocks

```markdown
# Claude-specific formatting
{{#model claude}}
<instructions>
Follow these rules carefully.
</instructions>
{{/model}}

# GPT-specific formatting  
{{#model gpt}}
[INSTRUCTIONS]
Follow these rules carefully.
[/INSTRUCTIONS]
{{/model}}
```

---

## 7. Guardrail File Format

```yaml
# guardrails/safety.yaml

name: safety
priority: 100

hard_constraints:
  - name: "constraint_name"
    description: "What this constraint prevents"
    rules:
      - "Rule 1"
      - "Rule 2"
    trigger_phrases:           # Optional: patterns that trigger this
      - "pattern 1"
      - "pattern 2"

soft_constraints:
  - name: "preference_name"
    description: "What this preference encourages"
    behavior: "gentle_redirect" # How to handle violations
    rules:
      - "Preference 1"

violation_responses:
  <constraint_name>:
    response: "Template response when violated"
    escalate: boolean
    log_level: "warn" | "error"

monitoring:
  log_violations: boolean
  alert_threshold: integer
  escalate_on:
    - "constraint_name"
```

---

## 8. Tool Definition Format

```yaml
# tools/search.yaml

name: search_products
version: "1.0.0"
description: "Search product catalog"

# OpenAI function calling format
openai_spec:
  type: function
  function:
    name: search_products
    description: "Search for products by name, category, or attributes"
    parameters:
      type: object
      properties:
        query:
          type: string
          description: "Search query"
        category:
          type: string
          enum: ["electronics", "clothing", "home"]
        max_results:
          type: integer
          default: 10
      required: ["query"]

# Response schema
response_schema:
  type: object
  properties:
    products:
      type: array
      items:
        $ref: "#/definitions/Product"

definitions:
  Product:
    type: object
    properties:
      id: { type: string }
      name: { type: string }
      price: { type: number }

# Usage constraints
constraints:
  rate_limit: 10           # Per session
  requires_auth: false
  cost_tier: "low"

# Example I/O for testing
examples:
  - input: { query: "laptop" }
    output: { products: [...] }
```

---

## 9. CLI Commands

### 9.1 Command Reference

```bash
# Package Management
cop init [name]              # Initialize new package
cop install [pkg] [--save]   # Install dependency
cop uninstall <pkg>          # Remove dependency
cop update [pkg]             # Update dependencies
cop list                     # List installed packages
cop outdated                 # Check for updates

# Development
cop lint                     # Lint prompt files
cop validate                 # Validate cop.yaml
cop test [suite]             # Run tests
cop bench                    # Run benchmarks
cop repl                     # Interactive session

# Building
cop build [--target <t>]     # Build for target
cop pack                     # Create distributable archive
cop export <format>          # Export to other formats

# Publishing
cop login                    # Authenticate
cop publish                  # Publish to registry
cop unpublish <pkg@ver>      # Remove from registry
cop deprecate <pkg@ver>      # Mark as deprecated

# Discovery
cop search <query>           # Search registry
cop info <pkg>               # Package details
cop docs <pkg>               # Open documentation
```

### 9.2 Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Package not found |
| 4 | Dependency conflict |
| 5 | Lint/validation failure |
| 6 | Test failure |
| 7 | Build failure |
| 8 | Publish failure |
| 9 | Authentication failure |

---

## 10. Registry API

### 10.1 Endpoints

```
GET    /v1/packages                    # List packages
GET    /v1/packages/{name}             # Package metadata
GET    /v1/packages/{name}/{version}   # Specific version
GET    /v1/packages/{name}/-/{tarball} # Download package
PUT    /v1/packages/{name}             # Publish package
DELETE /v1/packages/{name}/{version}   # Unpublish
POST   /v1/packages/{name}/deprecate   # Deprecate version

GET    /v1/search?q={query}            # Search packages
GET    /v1/users/{username}/packages   # User's packages

POST   /v1/auth/login                  # Authenticate
POST   /v1/auth/token                  # Generate API token
```

### 10.2 Package Metadata Response

```json
{
  "name": "customer-support-agent",
  "description": "Production-ready customer support context module",
  "dist-tags": {
    "latest": "1.2.0",
    "beta": "1.3.0-beta.1"
  },
  "versions": {
    "1.2.0": {
      "name": "customer-support-agent",
      "version": "1.2.0",
      "dependencies": {
        "tone-analyzer": "^2.0.0"
      },
      "dist": {
        "tarball": "https://registry.cop.dev/.../1.2.0.tgz",
        "shasum": "abc123...",
        "integrity": "sha512-..."
      },
      "eval_results": {
        "gpt-4": { "score": 0.94, "date": "2025-01-15" },
        "claude-3": { "score": 0.91, "date": "2025-01-20" }
      }
    }
  },
  "maintainers": [
    { "name": "author", "email": "author@example.com" }
  ],
  "repository": { "type": "git", "url": "..." },
  "downloads": { "weekly": 1523, "monthly": 5891 }
}
```

---

## Appendix A: JSON Schema

The complete JSON Schema for `cop.yaml` validation is available at:
- https://cop.dev/schema/cop-manifest-v1.json

## Appendix B: Reserved Package Names

The following names are reserved and cannot be used:
- `cop`, `core`, `cli`, `runtime`, `registry`
- `test`, `eval`, `lint`, `build`
- Names starting with `_` or `.`

## Appendix C: Security Considerations

1. **Package Signing**: All published packages must be signed
2. **Content Scanning**: Packages are scanned for sensitive data
3. **Rate Limiting**: API calls are rate-limited per user/token
4. **Audit Logging**: All publish/unpublish actions are logged

---

*Specification Status: DRAFT*
*Last Updated: December 2025*
*Feedback: https://github.com/cop-lang/spec/issues*
