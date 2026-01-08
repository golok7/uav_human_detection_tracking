import sys
import subprocess

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 11

def check_python_version():
    major, minor = sys.version_info[:2]
    if major != REQUIRED_MAJOR or minor != REQUIRED_MINOR:
        print(f"❌ Python {REQUIRED_MAJOR}.{REQUIRED_MINOR} required.")
        print(f"   Current version: Python {major}.{minor}")
        print("   Please install Python 3.11.x")
        sys.exit(1)
    print(f"✅ Python version OK: {major}.{minor}")

def install_pytorch():
    print("📦 Installing PyTorch (CPU version)...")
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision", "torchaudio"
    ]
    subprocess.check_call(cmd)

def verify_installation():
    try:
        import torch
        print(f"✅ PyTorch installed successfully: {torch.__version__}")
    except ImportError:
        print("❌ PyTorch installation failed")

if __name__ == "__main__":
    check_python_version()
    install_pytorch()
    verify_installation()
