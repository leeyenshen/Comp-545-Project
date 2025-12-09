#!/usr/bin/env python3
"""
Environment Setup Check
Verifies that all dependencies are installed correctly
"""

import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    else:
        print("✅ Python version OK")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name

    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        print(f"❌ {package_name}: NOT INSTALLED")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return True
        else:
            print("⚠️  CUDA not available (CPU only)")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_java():
    """Check Java installation (for Pyserini)"""
    import subprocess
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        version_line = result.stderr.split('\n')[0]
        print(f"✅ Java: {version_line}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Java not found (required for Pyserini/BM25)")
        return False

def check_directories():
    """Check if project directories exist"""
    project_root = Path(__file__).parent
    required_dirs = [
        'config',
        'data',
        'src',
        'scripts',
        'outputs',
        'paper'
    ]

    all_exist = True
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ exists")
        else:
            print(f"❌ {dir_name}/ missing")
            all_exist = False

    return all_exist

def main():
    """Main check function"""
    print("="*60)
    print("RAG HALLUCINATION DETECTION - ENVIRONMENT CHECK")
    print("="*60)

    all_ok = True

    # Check Python version
    print("\n1. Python Version")
    print("-"*60)
    all_ok &= check_python_version()

    # Check core packages
    print("\n2. Core ML/NLP Libraries")
    print("-"*60)
    packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('datasets', 'datasets'),
        ('sentence-transformers', 'sentence_transformers'),
    ]

    for pkg_name, import_name in packages:
        all_ok &= check_package(pkg_name, import_name)

    # Check retrieval packages
    print("\n3. Retrieval Libraries")
    print("-"*60)
    retrieval_packages = [
        ('pyserini', 'pyserini'),
        ('faiss-cpu', 'faiss'),
    ]

    for pkg_name, import_name in retrieval_packages:
        all_ok &= check_package(pkg_name, import_name)

    # Check detection packages
    print("\n4. Detection Libraries")
    print("-"*60)
    detection_packages = [
        ('ragas', 'ragas'),
        ('scikit-learn', 'sklearn'),
    ]

    for pkg_name, import_name in detection_packages:
        all_ok &= check_package(pkg_name, import_name)

    # Check visualization packages
    print("\n5. Visualization Libraries")
    print("-"*60)
    viz_packages = [
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
    ]

    for pkg_name, import_name in viz_packages:
        all_ok &= check_package(pkg_name, import_name)

    # Check utilities
    print("\n6. Utility Libraries")
    print("-"*60)
    util_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
        ('nltk', 'nltk'),
        ('pyyaml', 'yaml'),
    ]

    for pkg_name, import_name in util_packages:
        all_ok &= check_package(pkg_name, import_name)

    # Check CUDA
    print("\n7. CUDA/GPU")
    print("-"*60)
    check_cuda()  # Don't require CUDA

    # Check Java
    print("\n8. Java (for BM25)")
    print("-"*60)
    check_java()  # Don't require Java initially

    # Check directories
    print("\n9. Project Structure")
    print("-"*60)
    all_ok &= check_directories()

    # Download NLTK data
    print("\n10. NLTK Data")
    print("-"*60)
    try:
        import nltk
        nltk_packages = ['stopwords', 'wordnet', 'punkt']
        for pkg in nltk_packages:
            try:
                nltk.data.find(f'corpora/{pkg}')
                print(f"✅ NLTK {pkg} downloaded")
            except LookupError:
                print(f"⚠️  NLTK {pkg} not found, downloading...")
                nltk.download(pkg, quiet=True)
                print(f"✅ NLTK {pkg} downloaded")
    except Exception as e:
        print(f"❌ Error with NLTK: {e}")

    # Summary
    print("\n" + "="*60)
    if all_ok:
        print("✅ ALL CHECKS PASSED - Ready to run pipeline!")
        print("\nNext steps:")
        print("  1. Review QUICKSTART.md for usage instructions")
        print("  2. Run: ./run_all.sh")
        print("  3. Or run step-by-step:")
        print("     python scripts/01_download_datasets.py")
        print("     python scripts/02_build_bm25_index.py")
        print("     python scripts/03_build_faiss_index.py")
        print("     python scripts/04_run_pipeline.py")
        print("     python scripts/05_create_visualizations.py")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nTo fix:")
        print("  1. Install missing packages:")
        print("     pip install -r requirements.txt")
        print("  2. Install Java 11+ for Pyserini:")
        print("     - macOS: brew install openjdk@11")
        print("     - Ubuntu: sudo apt-get install openjdk-11-jdk")
        print("  3. Re-run this check: python check_setup.py")

    print("="*60)

    return all_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
