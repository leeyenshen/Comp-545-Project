#!/bin/bash
# Attempt to fix BM25/Pyserini on macOS ARM
# This uses Rosetta 2 to run x86_64 Java

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     BM25/PYSERINI FIX FOR macOS ARM                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "⚠️  WARNING: This is experimental and may not work!"
echo "BM25 is NOT required for the pipeline to work."
echo ""
read -p "Continue anyway? (y/N) " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted. Pipeline will use FAISS-only retrieval."
    exit 0
fi

echo ""
echo "Method 1: Install x86_64 Java via Rosetta"
echo "════════════════════════════════════════════════════════════════"

# Check if Rosetta is installed
if ! pgrep oahd > /dev/null 2>&1; then
    echo "Installing Rosetta 2..."
    softwareupdate --install-rosetta --agree-to-license
fi

# Install x86_64 Homebrew
if [ ! -d "/usr/local/Homebrew" ]; then
    echo "Installing x86_64 Homebrew..."
    arch -x86_64 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi

# Install x86_64 Java
echo "Installing x86_64 Java..."
arch -x86_64 /usr/local/bin/brew install openjdk@21

# Set Java environment
export JAVA_HOME=/usr/local/opt/openjdk@21/libexec/openjdk.jdk/Contents/Home
export PATH=$JAVA_HOME/bin:$PATH
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib

echo ""
echo "Java installation:"
java -version
echo ""

# Reinstall Pyserini
echo "Reinstalling Pyserini..."
source venv/bin/activate
pip uninstall pyserini -y
pip install pyserini

# Test
echo ""
echo "Testing BM25 indexing..."
python3 scripts/02_build_bm25_index.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ BM25 working!"
else
    echo ""
    echo "❌ BM25 still failing. Using FAISS-only is recommended."
fi
