#!/bin/bash
# Run BM25 indexing with correct Java environment

# Set Java environment variables
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@21/21.0.9/libexec/openjdk.jdk/Contents/Home
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib
export PATH="$JAVA_HOME/bin:$PATH"
export DYLD_LIBRARY_PATH="$JAVA_HOME/lib/server:$DYLD_LIBRARY_PATH"

echo "Running BM25 indexing with Java 21..."
echo "JAVA_HOME: $JAVA_HOME"
echo "JVM_PATH: $JVM_PATH"
echo ""

# Run the BM25 indexing script
python scripts/02_build_bm25_index.py
