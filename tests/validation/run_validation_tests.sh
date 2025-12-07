#!/bin/bash
# Run validation tests for COP schema validation
# Uses the cop CLI validate command

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

# Check if cop CLI is available
if ! command -v cop &> /dev/null; then
    echo -e "${YELLOW}cop CLI not found, trying python -m cop${NC}"
    COP_CMD="python -m cop"
else
    COP_CMD="cop"
fi

# Test 1: Valid packages should pass
echo -e "${YELLOW}Test 1: Validating customer-support-agent (should pass)${NC}"
if $COP_CMD validate examples/customer-support-agent 2>/dev/null; then
    echo -e "${GREEN}✓ PASSED: customer-support-agent validation succeeded${NC}"
else
    echo -e "${RED}✗ FAILED: customer-support-agent should pass validation${NC}"
fi
echo ""

# Test 2: Validate playwright-test-agent
echo -e "${YELLOW}Test 2: Validating playwright-test-agent (should pass)${NC}"
if $COP_CMD validate examples/playwright-test-agent 2>/dev/null; then
    echo -e "${GREEN}✓ PASSED: playwright-test-agent validation succeeded${NC}"
else
    echo -e "${RED}✗ FAILED: playwright-test-agent should pass validation${NC}"
fi
echo ""

# Test 3: Summary of all examples
echo -e "${YELLOW}Test 3: Validating all examples${NC}"
echo ""

for dir in examples/*/; do
    if [ -f "${dir}cop.yaml" ]; then
        name=$(basename "$dir")
        if $COP_CMD validate "$dir" 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} $name"
        else
            echo -e "  ${RED}✗${NC} $name"
        fi
    fi
done

echo ""
echo "=========================================="
echo "Validation tests completed"
echo "=========================================="
