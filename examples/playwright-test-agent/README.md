# Playwright Test Agent - COP Example

This is an example COP package demonstrating an **AI coding agent specialized in generating Playwright end-to-end tests** for web applications.

## Overview

The Playwright Test Agent helps developers create, maintain, and improve Playwright tests by:

- Generating high-quality test code following best practices
- Understanding project structure and existing test patterns
- Providing guidance on testing concepts and Playwright patterns
- Ensuring tests are reliable, maintainable, and secure

## Package Structure

```
playwright-test-agent/
├── cop.yaml                    # Package manifest
├── README.md                   # This file
│
├── prompts/
│   └── system.md              # Core system prompt for the agent
│
├── personas/
│   ├── technical.yaml         # Precise, code-focused communication
│   ├── helpful.yaml           # Friendly, educational approach
│   └── concise.yaml           # Brief, direct responses
│
├── guardrails/
│   ├── safety.yaml            # Core safety constraints
│   ├── code-quality.yaml      # Code quality enforcement
│   └── security.yaml          # Security considerations
│
├── tools/
│   ├── read_file.yaml         # Read and analyze source files
│   ├── analyze_project_structure.yaml  # Understand project layout
│   ├── search_codebase.yaml   # Search for patterns/components
│   ├── check_existing_tests.yaml      # Review existing test patterns
│   └── validate_test_code.yaml        # Validate generated code
│
├── knowledge/
│   ├── best-practices.md      # Playwright testing best practices
│   ├── common-patterns.md    # Common test patterns and examples
│   └── selectors-guide.md     # Selector strategy guide
│
└── tests/
    ├── behavioral/
    │   └── quality.yaml       # Quality evaluation tests
    └── safety/
        └── security.yaml      # Security and safety tests
```

## Key Features

### 1. Intelligent Code Generation

The agent generates Playwright tests that:
- Use semantic selectors (`getByRole()`, `getByLabel()`, etc.)
- Follow best practices (proper waiting, assertions, structure)
- Match existing project patterns
- Are maintainable and reliable

### 2. Project Awareness

Tools enable the agent to:
- Understand project structure and framework
- Analyze existing test patterns
- Search codebase for components and functionality
- Maintain consistency with existing tests

### 3. Multiple Personas

Choose a communication style:
- **Technical**: Precise, code-focused for experienced developers
- **Helpful**: Friendly, educational with explanations
- **Concise**: Brief, direct responses focused on code

### 4. Safety & Quality Guardrails

Three layers of guardrails ensure:
- **Safety**: No hardcoded credentials, production modifications, or dangerous operations
- **Code Quality**: Best practices, proper structure, reliable selectors
- **Security**: Secure authentication, input validation, no secrets in code

### 5. Comprehensive Knowledge Base

Built-in knowledge includes:
- Playwright best practices
- Common test patterns
- Selector strategies
- Error handling approaches

## Usage Example

```yaml
# cop.yaml configuration
context:
  system:
    source: "./prompts/system.md"
    variables:
      project_name: "My Web App"
      project_type: "spa"
      language: "typescript"
      base_url: "https://staging.example.com"
      test_directory: "tests/e2e"
```

The agent can then:
1. Analyze your project structure
2. Review existing test patterns
3. Generate new tests matching your conventions
4. Validate code quality and best practices

## Example Interactions

### Generate Login Test

**User**: "Write a test for the login form"

**Agent** (Technical persona):
```typescript
import { test, expect } from '@playwright/test';

test.describe('Login Form', () => {
  test('should successfully log in with valid credentials', async ({ page }) => {
    await page.goto('/login');
    await page.getByLabel('Email').fill('user@example.com');
    await page.getByLabel('Password').fill('password123');
    await page.getByRole('button', { name: 'Sign in' }).click();
    await expect(page).toHaveURL(/dashboard/);
  });
});
```

Uses `getByLabel()` for accessible form inputs and `getByRole()` for the button. The URL assertion verifies successful navigation after login.

### Generate Test with API Wait

**User**: "Write a test that waits for an API call before asserting"

**Agent**:
```typescript
test('should wait for API response', async ({ page }) => {
  await page.goto('/dashboard');
  
  const responsePromise = page.waitForResponse(
    response => response.url().includes('/api/data') && response.status() === 200
  );
  
  await page.getByRole('button', { name: 'Load Data' }).click();
  await responsePromise;
  
  await expect(page.getByTestId('data-table')).toBeVisible();
});
```

Uses `waitForResponse()` to wait for the API call instead of arbitrary timeouts.

## Configuration Variables

| Variable | Type | Required | Description |
|----------|------|----------|-------------|
| `project_name` | string | Yes | Name of the project being tested |
| `project_type` | string | No | Type: web-application, spa, ssr, etc. |
| `test_framework` | string | No | Framework variant (default: playwright) |
| `language` | string | No | TypeScript or JavaScript (default: typescript) |
| `base_url` | string | No | Base URL of the application |
| `test_directory` | string | No | Where tests are stored (default: tests/e2e) |

## Tools Available

1. **read_file**: Read source files to understand components and functionality
2. **analyze_project_structure**: Understand project organization and framework
3. **search_codebase**: Find components, functions, and patterns
4. **check_existing_tests**: Review existing test patterns for consistency
5. **validate_test_code**: Validate generated code for syntax and best practices

## Evaluation

The package includes comprehensive test suites:

- **Behavioral Tests**: Evaluate code quality, correctness, best practices adherence
- **Security Tests**: Ensure no hardcoded credentials, dangerous operations, or security vulnerabilities

## Best Practices Enforced

- ✅ Semantic selectors (getByRole, getByLabel) over CSS/XPath
- ✅ Proper waiting strategies (no arbitrary timeouts)
- ✅ Explicit assertions with expect()
- ✅ Test isolation and independence
- ✅ Page Object Model for complex applications
- ✅ Environment variables for credentials
- ✅ Test data cleanup

## Comparison with Customer Support Agent

| Aspect | Customer Support Agent | Playwright Test Agent |
|--------|----------------------|----------------------|
| **Domain** | Conversational AI | Code Generation |
| **Output** | Natural language responses | Test code |
| **Tools** | Order lookup, ticket creation | File reading, code analysis |
| **Guardrails** | Safety, compliance, brand | Safety, code quality, security |
| **Knowledge** | FAQ, policies | Best practices, patterns |
| **Personas** | Professional, friendly | Technical, helpful, concise |

## Dependencies

- `code-analyzer`: For codebase analysis
- `test-patterns`: For test pattern recognition

## License

MIT (same as parent repository)

---

This example demonstrates how COP packages can be specialized for specific domains (testing, support, etc.) while maintaining the same structure and principles.
