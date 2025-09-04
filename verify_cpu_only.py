#!/usr/bin/env python3
"""
CPU-Only Verification Script
This script verifies that only CPU versions are installed - NO GPU/CUDA dependencies
"""

import sys
import subprocess

def check_pytorch_cpu():
    """Check PyTorch installation is CPU-only."""
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        
        if "+cpu" not in torch.__version__:
            print("❌ WARNING: PyTorch version doesn't indicate CPU-only build")
            return False
            
        if torch.cuda.is_available():
            print("❌ ERROR: CUDA is available - GPU dependencies detected!")
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            return False
        else:
            print("✅ CUDA not available - CPU-only confirmed")
            return True
            
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_no_cuda_packages():
    """Check that no CUDA packages are installed."""
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Failed to get package list")
            return False
            
        packages = result.stdout.lower()
        cuda_packages = []
        
        # Check for common CUDA package names
        cuda_keywords = ['cuda', 'cudnn', 'cublas', 'cufft', 'curand', 'cusparse', 'cusolver']
        for keyword in cuda_keywords:
            if keyword in packages:
                cuda_packages.append(keyword)
        
        if cuda_packages:
            print(f"❌ CUDA-related packages found: {cuda_packages}")
            return False
        else:
            print("✅ No CUDA-related packages found")
            return True
            
    except Exception as e:
        print(f"❌ Error checking packages: {e}")
        return False

def check_pipeline_imports():
    """Check that both pipelines can be imported."""
    try:
        from text_to_image_pipeline import TextToImagePipeline
        from text_to_image_pipeline_lite import TextToImagePipelineLite
        print("✅ Both pipelines import successfully")
        return True
    except ImportError as e:
        print(f"❌ Pipeline import error: {e}")
        return False

def main():
    """Main verification process."""
    print("🔍 Verifying CPU-only installation...")
    print("=" * 50)
    
    checks = [
        ("PyTorch CPU-only", check_pytorch_cpu),
        ("No CUDA packages", check_no_cuda_packages),
        ("Pipeline imports", check_pipeline_imports)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All checks passed! CPU-only installation verified.")
        print("✅ You can safely use both lite and full pipelines.")
    else:
        print("❌ Some checks failed! GPU dependencies may be present.")
        print("💡 Run the CPU-only installation script to fix issues.")
        sys.exit(1)

if __name__ == "__main__":
    main()
