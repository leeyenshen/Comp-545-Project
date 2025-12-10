# OpenAI API Setup for RAGAS - Quick Guide

## Why Use OpenAI API?

| Approach | Time | Cost | Accuracy |
|----------|------|------|----------|
| Local TinyLlama | 45-60 min | Free | Lower (small model) |
| **OpenAI API** | **5-10 min** | **~$0.10-0.50** | **Higher (GPT-3.5)** |

## Step 1: Get OpenAI API Key

1. Go to: https://platform.openai.com/api-keys
2. Sign up / Log in
3. Click "Create new secret key"
4. Copy the key (starts with `sk-...`)

**Cost Note:**
- First time users get $5 free credit
- This experiment uses ~$0.10-0.50 total
- RAGAS will use GPT-3.5-turbo (cheapest, fast)

## Step 2: Run Setup Script

```bash
./setup_openai.sh
```

This will:
- Prompt you to paste your API key
- Save it to `.env` file
- Export it for current session
- Add to your shell config (~/.zshrc or ~/.bashrc)

## Step 3: Run Detectors with OpenAI

```bash
./run_detectors_openai.sh
```

This will:
- Load your existing QA results
- Run RAGAS with OpenAI API (fast!)
- Run NLI and Lexical locally
- Save updated results

**Expected time: 5-10 minutes** ⚡

## Alternative: Set API Key Manually

If you prefer not to use the setup script:

```bash
# Temporary (current session only)
export OPENAI_API_KEY='sk-your-key-here'

# Then run detectors
./run_detectors_openai.sh
```

Or create `.env` file manually:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
./run_detectors_openai.sh
```

## What Gets Called?

The script uses:
- **Model**: `gpt-3.5-turbo` (default for RAGAS)
- **Requests**: ~150-200 API calls total (3 tiers × 50 samples)
- **Tokens**: ~50,000-100,000 tokens
- **Cost**: $0.10-0.50 (at $0.002 per 1K tokens)

## Troubleshooting

### "Invalid API key"
- Check the key starts with `sk-`
- Make sure you copied the full key
- Create a new key if needed

### "Insufficient quota"
- You need billing set up on OpenAI account
- Add $5 minimum to your account
- First-time users get $5 free credit

### "Rate limit exceeded"
- Wait a few seconds and re-run
- The script will continue from where it failed

### "Module openai not found"
```bash
source venv/bin/activate
pip install openai
```

## Security Note

⚠️ **Never commit `.env` to git!**

The `.gitignore` file should already exclude it, but check:
```bash
grep ".env" .gitignore  # Should show .env is ignored
```

## After Running

Check your results:
```bash
# See RAGAS scores
head -2 outputs/results/results_high.csv

# Should show filled columns:
# ragas_hallucinated, ragas_faithfulness
```

Then generate visualizations:
```bash
python scripts/05_create_visualizations.py
```

## Full Workflow

```bash
# 1. Get API key from OpenAI
# 2. Run setup
./setup_openai.sh

# 3. Run detectors (5-10 min)
./run_detectors_openai.sh

# 4. Generate visualizations
python scripts/05_create_visualizations.py

# 5. Check results
open outputs/visualizations/
```

**Total time: ~15 minutes for complete results!** 🎉
