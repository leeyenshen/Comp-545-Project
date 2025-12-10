#!/usr/bin/env python3
"""
Fix NLTK data download for macOS SSL certificate issues
"""

import ssl
import nltk

# Bypass SSL certificate verification (macOS issue)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
print("Downloading NLTK data...")
nltk.download('stopwords', quiet=False)
nltk.download('wordnet', quiet=False)
nltk.download('punkt', quiet=False)
nltk.download('omw-1.4', quiet=False)

print("\n✅ NLTK data downloaded successfully!")
