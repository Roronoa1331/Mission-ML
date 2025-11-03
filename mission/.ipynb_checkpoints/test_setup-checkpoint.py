"""
Setup Verification Script
Tests if all dependencies are installed correctly
"""

import sys
import os

def test_imports():
    """测试所有必需的包 / Test all required packages"""
    packages = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'skimage': 'scikit-image',
        'lime': 'LIME'
    }
    
    print("\n📦 Testing package imports...")
    print("-" * 60)
    
    failed = []
    versions = {}
    
    for package, name in packages.items():
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            versions[package] = version
            print(f"✓ {name:<20} (version: {version})")
        except ImportError as e:
            print(f"✗ {name:<20} FAILED: {e}")
            failed.append(name)
    
    print("-" * 60)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("\n💡 Solution: Run the following command:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_tensorflow():
    """测试TensorFlow / Test TensorFlow"""
    print("\n🔧 Testing TensorFlow...")
    print("-" * 60)
    
    try:
        import tensorflow as tf
        
        # Version
        print(f"TensorFlow version: {tf.__version__}")
        
        # GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ GPU available: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
        else:
            print("⚠ No GPU detected (will use CPU)")
        
        # Simple computation test
        print("\nTesting TensorFlow computation...")
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        
        if c.shape == (2, 2):
            print("✓ TensorFlow computation test: PASS")
            print("-" * 60)
            return True
        else:
            print("✗ TensorFlow computation test: FAIL")
            print("-" * 60)
            return False
            
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        print("-" * 60)
        return False


def test_dataset():
    """测试数据集 / Test dataset"""
    print("\n📁 Testing dataset...")
    print("-" * 60)
    
    dataset_path = "./casting_512x512/"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\n💡 Solution:")
        print("   1. Download dataset from Kaggle:")
        print("      https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product")
        print("   2. Extract to: ./casting_512x512/")
        print("-" * 60)
        return False
    
    # Check subdirectories
    def_path = os.path.join(dataset_path, "def_front")
    ok_path = os.path.join(dataset_path, "ok_front")
    
    if not os.path.exists(def_path):
        print(f"❌ Missing directory: {def_path}")
        print("-" * 60)
        return False
    
    if not os.path.exists(ok_path):
        print(f"❌ Missing directory: {ok_path}")
        print("-" * 60)
        return False
    
    # Count images
    def_images = [f for f in os.listdir(def_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
    ok_images = [f for f in os.listdir(ok_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
    
    def_count = len(def_images)
    ok_count = len(ok_images)
    total_count = def_count + ok_count
    
    print(f"✓ Dataset found at: {dataset_path}")
    print(f"\nDataset statistics:")
    print(f"  Defective images (def_front): {def_count}")
    print(f"  OK images (ok_front):         {ok_count}")
    print(f"  Total images:                 {total_count}")
    
    if total_count < 100:
        print("\n⚠ Warning: Dataset seems incomplete (too few images)")
        print("-" * 60)
        return False
    
    print("-" * 60)
    return True


def test_image_loading():
    """测试图像加载 / Test image loading"""
    print("\n🖼️  Testing image loading...")
    print("-" * 60)
    
    try:
        from tensorflow import keras
        import numpy as np
        
        dataset_path = "./casting_512x512/"
        
        if not os.path.exists(dataset_path):
            print("⚠ Skipping (dataset not found)")
            print("-" * 60)
            return True
        
        # Try to load one image
        def_path = os.path.join(dataset_path, "def_front")
        images = [f for f in os.listdir(def_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        
        if images:
            test_image = os.path.join(def_path, images[0])
            img = keras.preprocessing.image.load_img(test_image, target_size=(299, 299))
            img_array = keras.preprocessing.image.img_to_array(img)
            
            print(f"✓ Successfully loaded test image: {images[0]}")
            print(f"  Image shape: {img_array.shape}")
            print(f"  Image dtype: {img_array.dtype}")
            print("-" * 60)
            return True
        else:
            print("⚠ No images found to test")
            print("-" * 60)
            return True
            
    except Exception as e:
        print(f"❌ Image loading test failed: {e}")
        print("-" * 60)
        return False


def print_system_info():
    """打印系统信息 / Print system information"""
    print("\n💻 System Information")
    print("-" * 60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    
    try:
        import platform
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"Processor: {platform.processor()}")
    except:
        pass
    
    print("-" * 60)


def print_next_steps(success):
    """打印下一步操作 / Print next steps"""
    print("\n" + "=" * 60)
    
    if success:
        print("✅ SETUP VERIFICATION COMPLETE!")
        print("=" * 60)
        print("\n🎉 You're ready to start!")
        print("\n📚 Next steps:")
        print("   1. Quick start (recommended for beginners):")
        print("      python quick_start.py")
        print("\n   2. Full training pipeline:")
        print("      python casting_defect_detection.py")
        print("\n   3. Follow the lab manual:")
        print("      Open 实验手册.md for step-by-step instructions")
        print("\n   4. Make predictions:")
        print("      python predict.py --model model.h5 --image test.jpeg")
    else:
        print("❌ SETUP INCOMPLETE")
        print("=" * 60)
        print("\n⚠️  Please fix the issues above before proceeding.")
        print("\n💡 Common solutions:")
        print("   • Install dependencies: pip install -r requirements.txt")
        print("   • Download dataset from Kaggle")
        print("   • Check Python version (requires 3.7+)")
    
    print("\n" + "=" * 60)


def main():
    """主函数 / Main function"""
    print("=" * 60)
    print("CASTING DEFECT DETECTION - SETUP VERIFICATION")
    print("=" * 60)
    
    # Print system info
    print_system_info()
    
    # Run tests
    success = True
    
    # Test 1: Package imports
    success &= test_imports()
    
    # Test 2: TensorFlow
    success &= test_tensorflow()
    
    # Test 3: Dataset
    dataset_ok = test_dataset()
    # Don't fail overall if dataset is missing (can be downloaded later)
    if not dataset_ok:
        print("\n⚠️  Note: You can still proceed, but download the dataset before training")
    
    # Test 4: Image loading
    test_image_loading()
    
    # Print next steps
    print_next_steps(success)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

