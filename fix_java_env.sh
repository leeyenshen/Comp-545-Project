#!/bin/bash
# Fix Java environment for Pyserini

# Set JAVA_HOME to the actual OpenJDK installation
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@21/21.0.9/libexec/openjdk.jdk/Contents/Home

# Set JVM_PATH to point to libjvm.dylib
export JVM_PATH=$JAVA_HOME/lib/server/libjvm.dylib

# Add Java to PATH
export PATH="$JAVA_HOME/bin:$PATH"

# Set library path for dynamic linking (macOS)
export DYLD_LIBRARY_PATH="$JAVA_HOME/lib/server:$DYLD_LIBRARY_PATH"

echo "✓ Java environment configured:"
echo "  JAVA_HOME: $JAVA_HOME"
echo "  JVM_PATH: $JVM_PATH"
echo ""
java -version
