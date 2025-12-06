#!/bin/bash
# Run validation tests for COP schema validation
# This script demonstrates the schema validation capabilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"

echo "=========================================="
echo "COP Schema Validation Test Suite"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Test 1: Valid packages should pass
echo -e "${YELLOW}Test 1: Validating playwright-test-agent (should pass)${NC}"
if python3 validate_cop.py examples/playwright-test-agent 2>/dev/null; then
    echo -e "${GREEN}✓ PASSED: playwright-test-agent validation succeeded${NC}"
else
    echo -e "${RED}✗ FAILED: playwright-test-agent should pass validation${NC}"
fi
echo ""

# Test 2: Invalid files should fail
echo -e "${YELLOW}Test 2: Testing invalid files (should fail with errors)${NC}"
echo ""

for file in tests/validation/invalid-*.yaml; do
    filename=$(basename "$file")
    schema=""
    
    case "$filename" in
        invalid-manifest.yaml) schema="cop-manifest" ;;
        invalid-persona.yaml) schema="persona" ;;
        invalid-guardrail.yaml) schema="guardrail" ;;
        invalid-tool.yaml) schema="tool" ;;
        invalid-test.yaml) schema="test" ;;
    esac
    
    if [ -n "$schema" ]; then
        echo -e "  Testing ${filename}..."
        if python3 validate_cop.py --file "$file" --schema "$schema" 2>/dev/null; then
            echo -e "  ${RED}✗ UNEXPECTED: $filename should have failed validation${NC}"
        else
            errors=$(python3 validate_cop.py --file "$file" --schema "$schema" 2>&1 | grep -c "\[ERROR\]" || true)
            echo -e "  ${GREEN}✓ PASSED: $filename correctly detected $errors errors${NC}"
        fi
    fi
done
echo ""

# Test 3: Summary
echo -e "${YELLOW}Test 3: Full validation summary${NC}"
echo ""
python3 validate_cop.py --all 2>&1 | tail -20

echo ""
echo "=========================================="
echo "Validation tests completed"
echo "=========================================="
