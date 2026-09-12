#!/bin/bash
# ============================================================
# OS-Level Guardrails — One-Command Setup Script
# Run this INSIDE WSL2 Ubuntu:
#   chmod +x setup.sh && ./setup.sh
# ============================================================

set -e  # Stop on any error

echo "=========================================="
echo "  OS-Level Guardrails — Setup Script"
echo "=========================================="

# Step 1: Install system dependencies
echo ""
echo "[1/5] Installing system packages..."
sudo apt update -qq
sudo apt install -y -qq python3 python3-pip python3-venv gcc libseccomp-dev libseccomp2

# Step 2: Create Python virtual environment
echo ""
echo "[2/5] Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "  ✓ Virtual environment already exists"
fi

# Step 3: Activate and install Python deps
echo ""
echo "[3/5] Installing Python dependencies..."
source venv/bin/activate
pip install -q -r requirements.txt
echo "  ✓ Python packages installed"

# Step 4: Compile the seccomp filter
echo ""
echo "[4/5] Compiling seccomp filter..."
cd src/sandbox
make clean 2>/dev/null || true
make
cd ../..
echo "  ✓ Seccomp filter compiled"

# Step 5: Create logs directory
echo ""
echo "[5/5] Creating logs directory..."
mkdir -p logs
echo "  ✓ Logs directory ready"

echo ""
echo "=========================================="
echo "  ✅ Setup complete!"
echo ""
echo "  To activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "  To run the agent (UNSANDBOXED):"
echo "    python run_agent.py 'List files in /tmp'"
echo ""
echo "  To run the agent (SANDBOXED):"
echo "    python run_sandboxed.py 'List files in /tmp'"
echo ""
echo "  To start the dashboard:"
echo "    python -m src.dashboard.app"
echo "=========================================="
