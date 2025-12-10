#!/bin/bash
# Set up OpenAI API key for RAGAS

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     OPENAI API SETUP FOR RAGAS                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "You need an OpenAI API key to use RAGAS with OpenAI models."
echo ""
echo "Steps:"
echo "1. Go to: https://platform.openai.com/api-keys"
echo "2. Create a new API key (if you don't have one)"
echo "3. Copy the key"
echo "4. Enter it below"
echo ""
echo "Cost estimate: ~$0.10-0.50 for full experiment (very cheap!)"
echo ""

read -p "Enter your OpenAI API key: " api_key

if [ -z "$api_key" ]; then
    echo "❌ No API key entered. Exiting."
    exit 1
fi

# Export for current session
export OPENAI_API_KEY="$api_key"

# Save to .env file for persistence
echo "OPENAI_API_KEY=$api_key" > .env

# Add to bashrc/zshrc for future sessions
if [ -f ~/.zshrc ]; then
    echo "export OPENAI_API_KEY=$api_key" >> ~/.zshrc
elif [ -f ~/.bashrc ]; then
    echo "export OPENAI_API_KEY=$api_key" >> ~/.bashrc
fi

echo ""
echo "✅ OpenAI API key configured!"
echo ""
echo "Saved to:"
echo "  • Current session (exported)"
echo "  • .env file (for scripts)"
echo "  • Shell config (for future sessions)"
echo ""
echo "Next step: Run detectors with OpenAI"
echo "  ./run_detectors_openai.sh"
