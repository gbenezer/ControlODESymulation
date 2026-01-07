#!/bin/bash
# test_pyproject.sh - Comprehensive pyproject.toml testing script
# Tests installation, dependencies, and configuration for ControlDESymulation

# Don't exit on error - we want to collect all test results
set +e
set -o pipefail

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found in current directory"
    exit 1
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Test results array
declare -a FAILED_TESTS

# Helper functions
print_header() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

print_test() {
    echo -e "${YELLOW}→ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
    ((TESTS_PASSED++))
}

print_failure() {
    echo -e "${RED}✗ $1${NC}"
    ((TESTS_FAILED++))
    FAILED_TESTS+=("$1")
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

run_test() {
    ((TESTS_RUN++))
}

# Cleanup function
cleanup() {
    print_info "Cleaning up test environments..."
    rm -rf test_venv_*
    rm -rf dist/ build/ *.egg-info 2>/dev/null
    echo ""
}

# Register cleanup on exit
trap cleanup EXIT INT TERM

# =============================================================================
# TEST 1: TOML Syntax Validation
# =============================================================================
print_header "TEST 1: TOML Syntax Validation"
run_test

print_test "Validating pyproject.toml syntax..."

# Try tomllib first (Python 3.11+)
python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null
if [ $? -eq 0 ]; then
    print_success "pyproject.toml has valid TOML syntax (via tomllib)"
else
    # Fall back to toml package
    print_test "tomllib not available, trying toml package..."
    python3 -m venv test_venv_toml_check >/dev/null 2>&1
    source test_venv_toml_check/bin/activate
    pip install toml -q >/dev/null 2>&1
    
    python3 -c "import toml; toml.load('pyproject.toml')" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "pyproject.toml has valid TOML syntax (via toml)"
    else
        print_failure "pyproject.toml has invalid TOML syntax"
    fi
    deactivate
    rm -rf test_venv_toml_check
fi

# =============================================================================
# TEST 2: Core Dependencies (NumPy Backend)
# =============================================================================
print_header "TEST 2: Core Dependencies Installation"
run_test

print_test "Creating clean virtual environment for core dependencies..."
python3 -m venv test_venv_core >/dev/null 2>&1
source test_venv_core/bin/activate

print_test "Installing core dependencies only (this may take a moment)..."
pip install -e . -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Core installation succeeded"
    
    print_test "Testing core imports..."
    python3 -c "import numpy; import sympy; import scipy; import pydantic; import typing_extensions" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "All core dependencies import successfully"
    else
        print_failure "Core dependency imports failed"
    fi
else
    print_failure "Core installation failed"
fi

deactivate

# =============================================================================
# TEST 3: PyTorch Backend
# =============================================================================
print_header "TEST 3: PyTorch Backend Installation"
run_test

print_test "Creating clean virtual environment for PyTorch backend..."
python3 -m venv test_venv_torch >/dev/null 2>&1
source test_venv_torch/bin/activate

print_test "Installing torch extra (this will take several minutes)..."
pip install -e ".[torch]" -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "PyTorch backend installation succeeded"
    
    print_test "Testing PyTorch imports..."
    TORCH_VERSION=$(python3 -c "import torch; import torchdiffeq; import torchsde; print(torch.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        print_success "PyTorch backend imports successfully (version: $TORCH_VERSION)"
    else
        print_failure "PyTorch backend imports failed"
    fi
else
    print_failure "PyTorch backend installation failed"
fi

deactivate

# =============================================================================
# TEST 4: JAX Backend
# =============================================================================
print_header "TEST 4: JAX Backend Installation"
run_test

print_test "Creating clean virtual environment for JAX backend..."
python3 -m venv test_venv_jax >/dev/null 2>&1
source test_venv_jax/bin/activate

print_test "Installing jax extra (this will take several minutes)..."
pip install -e ".[jax]" -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "JAX backend installation succeeded"
    
    print_test "Testing JAX imports..."
    JAX_VERSION=$(python3 -c "import jax; import jaxlib; import jaxtyping; import equinox; import diffrax; import optimistix; import lineax; print(jax.__version__)" 2>/dev/null)
    if [ $? -eq 0 ]; then
        print_success "JAX backend imports successfully (version: $JAX_VERSION)"
    else
        print_failure "JAX backend imports failed"
    fi
else
    print_failure "JAX backend installation failed"
fi

deactivate

# =============================================================================
# TEST 5: Visualization Tools
# =============================================================================
print_header "TEST 5: Visualization Tools"
run_test

print_test "Creating clean virtual environment for visualization..."
python3 -m venv test_venv_viz >/dev/null 2>&1
source test_venv_viz/bin/activate

print_test "Installing viz extra..."
pip install -e ".[viz]" -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Visualization tools installation succeeded"
    
    print_test "Testing visualization imports..."
    python3 -c "import plotly; import matplotlib" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Visualization tools import successfully"
    else
        print_failure "Visualization tools imports failed"
    fi
else
    print_failure "Visualization tools installation failed"
fi

deactivate

# =============================================================================
# TEST 6: Development Tools
# =============================================================================
print_header "TEST 6: Development Tools"
run_test

print_test "Creating clean virtual environment for dev tools..."
python3 -m venv test_venv_dev >/dev/null 2>&1
source test_venv_dev/bin/activate

print_test "Installing dev extra (this may take a moment)..."
pip install -e ".[dev]" -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Development tools installation succeeded"
    
    print_test "Testing dev tool availability..."
    TOOLS_OK=true
    
    if command -v pytest &> /dev/null; then
        PYTEST_VERSION=$(pytest --version 2>&1 | head -n1)
        print_info "  pytest: $PYTEST_VERSION"
    else
        TOOLS_OK=false
    fi
    
    if command -v black &> /dev/null; then
        BLACK_VERSION=$(black --version 2>&1)
        print_info "  black: $BLACK_VERSION"
    else
        TOOLS_OK=false
    fi
    
    if command -v ruff &> /dev/null; then
        RUFF_VERSION=$(ruff --version 2>&1)
        print_info "  ruff: $RUFF_VERSION"
    else
        TOOLS_OK=false
    fi
    
    if command -v mypy &> /dev/null; then
        MYPY_VERSION=$(mypy --version 2>&1)
        print_info "  mypy: $MYPY_VERSION"
    else
        TOOLS_OK=false
    fi
    
    if [ "$TOOLS_OK" = true ]; then
        print_success "All development tools available"
    else
        print_failure "Some development tools missing"
    fi
else
    print_failure "Development tools installation failed"
fi

deactivate

# =============================================================================
# TEST 7: Documentation Tools
# =============================================================================
print_header "TEST 7: Documentation Tools"
run_test

print_test "Creating clean virtual environment for docs..."
python3 -m venv test_venv_docs >/dev/null 2>&1
source test_venv_docs/bin/activate

print_test "Installing docs extra..."
pip install -e ".[docs]" -q >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Documentation tools installation succeeded"
    
    print_test "Testing docs imports..."
    python3 -c "import sphinx; import sphinx_rtd_theme" 2>/dev/null
    if [ $? -eq 0 ]; then
        print_success "Documentation tools import successfully"
    else
        print_failure "Documentation tools imports failed"
    fi
else
    print_failure "Documentation tools installation failed"
fi

deactivate

# =============================================================================
# TEST 8: Package Building
# =============================================================================
print_header "TEST 8: Package Building"
run_test

print_test "Installing build tools..."
python3 -m venv test_venv_build >/dev/null 2>&1
source test_venv_build/bin/activate
pip install build twine -q >/dev/null 2>&1

print_test "Building distribution packages..."
python3 -m build >/dev/null 2>&1
if [ $? -eq 0 ]; then
    print_success "Package build succeeded"
    
    print_test "Checking distribution with twine..."
    twine check dist/* >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "Distribution packages are valid"
    else
        print_failure "Distribution validation failed"
    fi
else
    print_failure "Package build failed"
fi

deactivate

# =============================================================================
# TEST 9: Tool Configurations (if src/ exists)
# =============================================================================
if [ -d "src" ]; then
    print_header "TEST 9: Tool Configuration Validation"
    run_test
    
    python3 -m venv test_venv_tools >/dev/null 2>&1
    source test_venv_tools/bin/activate
    pip install -e ".[dev]" -q >/dev/null 2>&1
    
    print_test "Testing pytest configuration..."
    pytest --collect-only >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "pytest configuration is valid"
    else
        print_info "pytest configuration could not be validated (no tests found or collection failed)"
    fi
    
    print_test "Testing black configuration..."
    black --check src/ >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "Code is properly formatted with black"
    else
        print_info "black would reformat some files (not a failure)"
    fi
    
    print_test "Testing ruff configuration..."
    ruff check src/ >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        print_success "ruff found no issues"
    else
        print_info "ruff found some issues (not counted as test failure)"
    fi
    
    deactivate
else
    print_info "Skipping tool configuration tests (src/ directory not found)"
fi

# =============================================================================
# SUMMARY
# =============================================================================
print_header "TEST SUMMARY"

echo -e "Tests run:    ${BLUE}${TESTS_RUN}${NC}"
echo -e "Tests passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Tests failed: ${RED}${TESTS_FAILED}${NC}"

if [ ${TESTS_FAILED} -eq 0 ]; then
    echo -e "\n${GREEN}═══════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ All tests passed! pyproject.toml is properly configured.${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}\n"
    exit 0
else
    echo -e "\n${RED}═══════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Some tests failed:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "${RED}  - $test${NC}"
    done
    echo -e "${RED}═══════════════════════════════════════════════════════${NC}\n"
    exit 1
fi