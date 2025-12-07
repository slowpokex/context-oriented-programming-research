"""
COP Init Command - Initialize a new COP package
"""

from pathlib import Path

from rich.console import Console
from rich.panel import Panel

# Package templates
TEMPLATES = {
    "basic": {
        "cop.yaml": '''# COP Package Manifest
meta:
  name: "{name}"
  version: "0.1.0"
  description: "A context-oriented programming package"

context:
  system:
    source: "./prompts/system.md"
  
  personas:
    default: "default"
    available:
      default:
        source: "./personas/default.yaml"
  
  guardrails:
    - name: safety
      source: "./guardrails/safety.yaml"
      priority: 100

  knowledge: []
  
  tools: []

runtime:
  model_config:
    temperature: 0.7
    max_tokens: 2048

evaluation:
  test_suites: []
''',
        "prompts/system.md": '''# {name} System Prompt

You are a helpful AI assistant.

## Role
Assist users with their questions and tasks.

## Guidelines
- Be helpful, harmless, and honest
- Provide clear and concise responses
- Ask for clarification when needed
- Respect user privacy
''',
        "personas/default.yaml": '''# Default Persona
name: default
description: "Default assistant persona"

tone: friendly
style: conversational

characteristics:
  - helpful
  - patient
  - clear

vocabulary:
  preferred:
    - "I'd be happy to help"
    - "Let me explain"
  avoided:
    - jargon
    - overly technical terms

response_patterns:
  greeting: "Hello! How can I assist you today?"
  clarification: "Could you please tell me more about...?"
  completion: "Is there anything else I can help with?"
''',
        "guardrails/safety.yaml": '''# Safety Guardrails
name: safety
description: "Core safety constraints"
priority: 100

hard_constraints:
  - id: no-harmful-content
    description: "Never generate harmful, illegal, or unethical content"
    trigger_patterns:
      - "how to harm"
      - "illegal activity"
    action: refuse
    response: "I can't help with that request."

  - id: no-personal-data
    description: "Never share or request sensitive personal information"
    trigger_patterns:
      - "social security"
      - "credit card number"
    action: refuse
    response: "I can't help with sensitive personal information."

soft_constraints:
  - id: stay-on-topic
    description: "Keep responses relevant to the conversation"
    action: redirect

violation_responses:
  refuse: "I'm not able to help with that. Let me know if there's something else I can assist with."
  redirect: "Let's focus on how I can best help you today."
''',
    },
    "customer-support": {
        "cop.yaml": '''# Customer Support Agent - COP Package
meta:
  name: "{name}"
  version: "0.1.0"
  description: "AI-powered customer support agent"

context:
  system:
    source: "./prompts/system.md"
  
  personas:
    default: "friendly"
    available:
      friendly:
        source: "./personas/friendly.yaml"
      professional:
        source: "./personas/professional.yaml"
  
  guardrails:
    - name: safety
      source: "./guardrails/safety.yaml"
      priority: 100
    - name: compliance
      source: "./guardrails/compliance.yaml"
      priority: 90

  knowledge:
    - name: faq
      source: "./knowledge/faq.md"
      type: static
  
  tools:
    - name: lookup_order
      source: "./tools/lookup_order.yaml"

runtime:
  model_config:
    temperature: 0.7
    max_tokens: 2048

evaluation:
  test_suites:
    - path: "./tests/behavioral"
      type: llm-judged
''',
        "prompts/system.md": '''# {name} - Customer Support System Prompt

You are a customer support agent for our company.

## Role
Help customers with inquiries about orders, products, and services.

## Responsibilities
1. Answer customer questions accurately
2. Look up order information when needed
3. Process simple requests (refunds under $50)
4. Escalate complex issues to human agents

## Guidelines
- Always be polite and professional
- Verify customer identity before sharing order details
- Never share internal policies or procedures
- Escalate if the customer is upset or the issue is complex

## Escalation Triggers
- Refund requests over $50
- Complaints about employees
- Legal or compliance questions
- Technical issues you can't resolve
''',
        "personas/friendly.yaml": '''name: friendly
description: "Warm and approachable customer support persona"
tone: warm
style: conversational

characteristics:
  - empathetic
  - patient
  - helpful

vocabulary:
  preferred:
    - "I'd love to help"
    - "Great question!"
    - "Let me take care of that"
  avoided:
    - "per our policy"
    - "unfortunately"
''',
        "personas/professional.yaml": '''name: professional
description: "Formal and efficient customer support persona"
tone: formal
style: concise

characteristics:
  - efficient
  - precise
  - respectful

vocabulary:
  preferred:
    - "I can assist you with that"
    - "Allow me to check"
  avoided:
    - slang
    - casual expressions
''',
        "guardrails/safety.yaml": '''name: safety
description: "Core safety guardrails"
priority: 100

hard_constraints:
  - id: no-customer-data-leak
    description: "Never reveal other customers' information"
    action: refuse

  - id: verify-identity
    description: "Verify customer identity before sharing account details"
    action: require_verification

soft_constraints:
  - id: professional-tone
    description: "Maintain professional communication"
    action: rephrase
''',
        "guardrails/compliance.yaml": '''name: compliance
description: "Regulatory compliance guardrails"
priority: 90

hard_constraints:
  - id: gdpr-compliance
    description: "Comply with data protection regulations"
    action: comply

  - id: refund-limits
    description: "Escalate refunds over $50"
    trigger_patterns:
      - "refund"
      - "money back"
    action: escalate_if_over_limit
    parameters:
      limit: 50
''',
        "knowledge/faq.md": '''# Frequently Asked Questions

## Orders
**Q: How do I track my order?**
A: You can track your order using the tracking link in your confirmation email, or by providing your order number.

**Q: How long does shipping take?**
A: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.

## Returns
**Q: What is your return policy?**
A: We accept returns within 30 days of purchase for unused items in original packaging.

**Q: How do I start a return?**
A: Contact us with your order number and reason for return. We'll provide a prepaid shipping label.
''',
        "tools/lookup_order.yaml": '''name: lookup_order
description: "Look up order details by order ID or customer email"

parameters:
  type: object
  properties:
    order_id:
      type: string
      description: "The order ID to look up"
    customer_email:
      type: string
      description: "Customer email for verification"
  required:
    - order_id

returns:
  type: object
  properties:
    status:
      type: string
      enum: [pending, shipped, delivered, cancelled]
    items:
      type: array
    total:
      type: number
''',
    },
}


def run_init(
    name: str,
    template: str = "basic",
    output_dir: Path = None,
    console: Console = None
) -> bool:
    """Initialize a new COP package.
    
    Args:
        name: Package name
        template: Template to use (basic, customer-support, coding-assistant)
        output_dir: Directory to create package in
        console: Rich console for output
    
    Returns:
        True if successful, False otherwise
    """
    console = console or Console()
    output_dir = output_dir or Path(".")
    
    # Get template
    if template not in TEMPLATES:
        console.print(f"[red]Error:[/] Unknown template: {template}")
        console.print(f"Available templates: {', '.join(TEMPLATES.keys())}")
        return False
    
    template_files = TEMPLATES[template]
    
    # Create package directory
    package_dir = output_dir / name
    
    if package_dir.exists():
        console.print(f"[red]Error:[/] Directory already exists: {package_dir}")
        return False
    
    console.print(Panel(
        f"Creating COP package: [bold]{name}[/]\nTemplate: [cyan]{template}[/]",
        title="[cyan]COP Init[/]",
        border_style="cyan"
    ))
    console.print()
    
    try:
        # Create directories
        dirs_to_create = [
            package_dir,
            package_dir / "prompts",
            package_dir / "personas",
            package_dir / "guardrails",
            package_dir / "knowledge",
            package_dir / "tools",
            package_dir / "tests" / "behavioral",
            package_dir / "tests" / "safety",
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"  [dim]Created:[/] {dir_path.relative_to(output_dir)}/")
        
        # Create files
        for file_path, content in template_files.items():
            full_path = package_dir / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Replace template variables
            content = content.replace("{name}", name)
            
            with open(full_path, "w") as f:
                f.write(content)
            
            console.print(f"  [green]Created:[/] {full_path.relative_to(output_dir)}")
        
        console.print()
        console.print("[bold green]✓ Package created successfully![/]")
        console.print()
        console.print("Next steps:")
        console.print(f"  1. [cyan]cd {name}[/]")
        console.print(f"  2. Edit [cyan]prompts/system.md[/] with your instructions")
        console.print(f"  3. Run [cyan]cop validate .[/] to check your package")
        console.print(f"  4. Run [cyan]cop build .[/] to build artifacts")
        console.print()
        
        return True
        
    except Exception as e:
        console.print(f"[red]Error creating package:[/] {e}")
        return False

