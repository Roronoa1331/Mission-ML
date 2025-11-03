# 环境配置指南 / Setup Guide

## 🚀 快速开始 / Quick Start

### 1️⃣ 检查Python版本 / Check Python Version

```bash
python --version
# 需要 Python 3.7 或更高版本 / Requires Python 3.7+
```

如果没有安装Python，请访问：https://www.python.org/downloads/

### 2️⃣ 创建虚拟环境（推荐）/ Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3️⃣ 安装依赖包 / Install Dependencies

```bash
pip install -r requirements.txt
```

如果安装速度慢，可以使用国内镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 4️⃣ 下载数据集 / Download Dataset

1. 访问 Kaggle: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
2. 下载数据集
3. 解压到项目目录，确保结构如下：

```
cam/
├── casting_512x512/
│   ├── def_front/
│   │   ├── cast_def_0_1.jpeg
│   │   ├── cast_def_0_2.jpeg
│   │   └── ...
│   └── ok_front/
│       ├── cast_ok_0_1.jpeg
│       ├── cast_ok_0_2.jpeg
│       └── ...
├── casting_defect_detection.py
├── quick_start.py
└── ...
```

### 5️⃣ 验证安装 / Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
```

应该看到类似输出：`TensorFlow version: 2.10.0` 或更高版本

### 6️⃣ 运行快速测试 / Run Quick Test

```bash
python quick_start.py
```

## 🔧 详细配置 / Detailed Configuration

### GPU支持 / GPU Support

#### 检查GPU是否可用 / Check GPU Availability

```python
import tensorflow as tf
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

#### NVIDIA GPU配置

1. **安装CUDA Toolkit** (推荐版本: 11.2)
   - 下载: https://developer.nvidia.com/cuda-toolkit-archive

2. **安装cuDNN** (推荐版本: 8.1)
   - 下载: https://developer.nvidia.com/cudnn

3. **验证GPU**
```bash
nvidia-smi
```

#### Apple Silicon (M1/M2) GPU配置

```bash
# 安装TensorFlow for macOS
pip install tensorflow-macos
pip install tensorflow-metal
```

### 常见问题解决 / Troubleshooting

#### ❌ 问题1: ImportError: No module named 'tensorflow'

**解决方案:**
```bash
pip install tensorflow>=2.10.0
```

#### ❌ 问题2: 内存不足 / Out of Memory

**解决方案:**
在脚本中修改batch_size:
```python
batch_size = 16  # 从64减小到16
```

#### ❌ 问题3: CUDA错误 / CUDA Error

**解决方案:**
```bash
# 使用CPU版本
pip uninstall tensorflow
pip install tensorflow-cpu
```

#### ❌ 问题4: 数据集路径错误

**解决方案:**
检查并修改脚本中的路径:
```python
dataset_url = "./casting_512x512/"  # 确保路径正确
```

## 📦 依赖包说明 / Package Details

| 包名 | 版本 | 用途 |
|------|------|------|
| tensorflow | >=2.10.0 | 深度学习框架 |
| keras | >=2.10.0 | 高级神经网络API |
| numpy | latest | 数值计算 |
| pandas | latest | 数据处理 |
| matplotlib | latest | 数据可视化 |
| Pillow | latest | 图像处理 |
| opencv-python | latest | 计算机视觉 |
| scikit-image | latest | 图像处理 |
| lime | latest | 模型解释 |

## 🌐 使用Google Colab（无需本地配置）

如果本地配置困难，可以使用Google Colab：

1. 访问: https://colab.research.google.com/
2. 创建新笔记本
3. 上传数据集到Google Drive
4. 运行以下代码：

```python
# 挂载Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安装依赖
!pip install lime

# 复制代码文件
# 上传 casting_defect_detection.py 到Colab

# 运行训练
!python casting_defect_detection.py
```

## 🐳 使用Docker（高级）

创建 `Dockerfile`:

```dockerfile
FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "casting_defect_detection.py"]
```

构建和运行:
```bash
docker build -t casting-detection .
docker run --gpus all -v $(pwd)/casting_512x512:/app/casting_512x512 casting-detection
```

## 📊 性能优化建议 / Performance Tips

### 1. 使用GPU加速
- 确保安装GPU版本的TensorFlow
- 使用 `tf.config.list_physical_devices('GPU')` 验证

### 2. 优化数据加载
```python
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
```

### 3. 混合精度训练（GPU）
```python
from tensorflow.keras import mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
```

### 4. 调整batch_size
- GPU内存充足: batch_size = 64 或更大
- GPU内存有限: batch_size = 32
- 仅CPU: batch_size = 16

## 🧪 测试安装 / Test Installation

创建测试脚本 `test_setup.py`:

```python
import sys

def test_imports():
    """测试所有必需的包"""
    packages = [
        'tensorflow',
        'keras',
        'numpy',
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'skimage',
        'lime'
    ]
    
    print("Testing package imports...")
    failed = []
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed.append(package)
    
    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        print("Please run: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True

def test_tensorflow():
    """测试TensorFlow"""
    import tensorflow as tf
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    
    # 简单计算测试
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"TensorFlow computation test: {'✓ PASS' if c.shape == (2, 2) else '✗ FAIL'}")

def test_dataset():
    """测试数据集"""
    import os
    dataset_path = "./casting_512x512/"
    
    if os.path.exists(dataset_path):
        def_count = len(os.listdir(os.path.join(dataset_path, "def_front")))
        ok_count = len(os.listdir(os.path.join(dataset_path, "ok_front")))
        print(f"\n✓ Dataset found!")
        print(f"  Defective images: {def_count}")
        print(f"  OK images: {ok_count}")
        return True
    else:
        print(f"\n❌ Dataset not found at {dataset_path}")
        print("Please download and extract the dataset.")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SETUP VERIFICATION")
    print("=" * 60)
    
    success = True
    success &= test_imports()
    
    try:
        test_tensorflow()
    except Exception as e:
        print(f"❌ TensorFlow test failed: {e}")
        success = False
    
    success &= test_dataset()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ Setup complete! You're ready to start.")
        print("\nNext steps:")
        print("  1. Run: python quick_start.py")
        print("  2. Or follow the lab manual: 实验手册.md")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
    print("=" * 60)
```

运行测试:
```bash
python test_setup.py
```

## 📚 额外资源 / Additional Resources

- **TensorFlow官方教程**: https://www.tensorflow.org/tutorials
- **Keras文档**: https://keras.io/
- **数据集来源**: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product
- **问题反馈**: 在GitHub Issues中提问

---

**配置完成后，开始你的深度学习之旅！** 🚀

