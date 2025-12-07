"""Validate JSONL dataset for fine-tuning compatibility."""
import json
import sys
from pathlib import Path


def validate_dataset(filepath: str) -> dict:
    """Validate a JSONL dataset file for fine-tuning."""
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}
    
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    results = {
        "total_lines": len(lines),
        "valid_examples": 0,
        "issues": [],
        "warnings": [],
    }
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        try:
            data = json.loads(line)
            
            # Check if it's metadata (not a training example)
            if data.get("_type") == "metadata":
                results["warnings"].append(f"Line {i}: Metadata entry (skip for fine-tuning)")
                continue
            
            # Check for messages array
            if "messages" not in data:
                results["issues"].append(f"Line {i}: Missing 'messages' array")
                continue
            
            messages = data["messages"]
            
            # Check message structure
            roles = [m.get("role") for m in messages]
            
            if "system" not in roles:
                results["issues"].append(f"Line {i}: Missing system message")
            if "user" not in roles:
                results["issues"].append(f"Line {i}: Missing user message")
            if "assistant" not in roles:
                results["issues"].append(f"Line {i}: Missing assistant message")
            
            # Check for empty content
            for m in messages:
                if not m.get("content"):
                    results["issues"].append(f"Line {i}: Empty {m.get('role')} content")
            
            # Check for extra fields
            extra_fields = [k for k in data.keys() if k not in ["messages"]]
            if extra_fields:
                results["warnings"].append(f"Line {i}: Extra fields {extra_fields}")
            
            # Check for undefined variables
            system_msg = next((m for m in messages if m["role"] == "system"), None)
            if system_msg and "{{UNDEFINED:" in system_msg.get("content", ""):
                results["warnings"].append(f"Line {i}: Unresolved template variables")
            
            results["valid_examples"] += 1
            
        except json.JSONDecodeError as e:
            results["issues"].append(f"Line {i}: Invalid JSON - {e}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset.jsonl>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    results = validate_dataset(filepath)
    
    print("=" * 50)
    print("JSONL Fine-tuning Validation Report")
    print("=" * 50)
    print()
    print(f"File: {filepath}")
    print(f"Total lines: {results['total_lines']}")
    print(f"Valid examples: {results['valid_examples']}")
    print()
    
    if results["issues"]:
        print("ERRORS (must fix):")
        for issue in results["issues"]:
            print(f"  ✗ {issue}")
        print()
    
    if results["warnings"]:
        print("WARNINGS (review):")
        for warning in results["warnings"]:
            print(f"  ⚠ {warning}")
        print()
    
    print("=" * 50)
    print("Fine-tuning Compatibility")
    print("=" * 50)
    print()
    
    if results["valid_examples"] > 0 and not results["issues"]:
        print("✓ Format: Valid OpenAI chat format")
        print("✓ Structure: system/user/assistant messages present")
    else:
        print("✗ Issues found - fix before fine-tuning")
    
    print()
    print(f"Examples: {results['valid_examples']} (minimum recommended: 10-50)")
    
    # Recommendations
    print()
    print("Recommendations:")
    if any("Metadata" in w for w in results["warnings"]):
        print("  1. Remove metadata line before fine-tuning")
    if any("Unresolved" in w for w in results["warnings"]):
        print("  2. Replace {{UNDEFINED:*}} with actual values")
    if any("Extra fields" in w for w in results["warnings"]):
        print("  3. Extra fields (_metadata) are usually ignored by fine-tuning")
    if results["valid_examples"] < 10:
        print("  4. Generate more examples (at least 10-50 recommended)")

