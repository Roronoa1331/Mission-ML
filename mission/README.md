# Casting Defect Detection using Deep Learning

A comprehensive deep learning project for industrial quality control, implementing binary image classification to detect defects in casting products.

## 📋 Project Overview

This project demonstrates:
- **Custom CNN Model**: Built from scratch for defect detection
- **Transfer Learning**: Using pre-trained Xception model
- **Data Augmentation**: Advanced image preprocessing techniques
- **Model Explainability**: Grad-CAM and LIME visualizations
- **Complete Pipeline**: From data exploration to model deployment

## 🎯 Features

- Binary classification (defective vs. non-defective casting products)
- Real-time data augmentation
- Efficient input pipelines using tf.data API
- Custom callbacks for training optimization
- Model interpretability with Grad-CAM and LIME
- Batch prediction capabilities
- Web application for easy deployment (optional)

## 📊 Dataset

- **Source**: Real-life Industrial Dataset of Casting Product
- **Image Size**: 512x512 pixels (resized to 299x299 for training)
- **Classes**: 
  - `ok_front`: Non-defective products
  - `def_front`: Defective products
- **Total Images**: ~1,300 samples

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.7+
TensorFlow 2.10+
GPU (recommended for faster training)
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle:
   - Dataset: "Real-life Industrial Dataset of Casting Product"
   - Extract to `./casting_512x512/`

4. Update the dataset path in `casting_defect_detection.py`:
```python
dataset_url = "./casting_512x512/"
```

### Training

Run the complete training pipeline:
```bash
python casting_defect_detection.py
```

This will:
1. Explore and visualize the dataset
2. Prepare and optimize data pipelines
3. Train a custom CNN model
4. Train a transfer learning model (Xception)
5. Save both models

### Prediction

Single image prediction:
```bash
python predict.py --model xception_transfer_model.h5 --image path/to/image.jpeg
```

Batch prediction:
```bash
python predict.py --model xception_transfer_model.h5 --batch path/to/image/folder/
```

### Web Application (Optional)

```bash
pip install flask
python app.py
```

Then open `http://localhost:5000` in your browser.

## 📁 Project Structure

```
cam/
├── casting_512x512/              # Dataset directory
│   ├── def_front/
│   └── ok_front/
├── casting_defect_detection.py  # Main training script
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── 实验手册.md                   # Lab manual (Chinese)
├── predict.py                    # Prediction script
├── app.py                        # Web application
├── custom_cnn_model.h5          # Saved custom model
└── xception_transfer_model.h5   # Saved transfer learning model
```

## 🧪 Experiments

The project includes 8 structured experiments for educational purposes:

1. **Data Exploration**: Understanding the dataset
2. **Data Preparation**: Creating train/validation splits
3. **Data Augmentation**: Implementing augmentation strategies
4. **Custom CNN**: Building and training a custom model
5. **Transfer Learning**: Using pre-trained Xception model
6. **Model Evaluation**: Comprehensive performance analysis
7. **Grad-CAM**: Visual explanations using Grad-CAM
8. **LIME**: Local interpretable model-agnostic explanations

See `实验手册.md` for detailed step-by-step instructions (in Chinese).

## 📈 Model Performance

### Custom CNN Model
- Training Accuracy: ~100%
- Validation Accuracy: ~100%
- Parameters: ~2.5M
- Training Time: ~2-3 minutes (with GPU)

### Transfer Learning (Xception)
- Training Accuracy: ~100%
- Validation Accuracy: ~100%
- Parameters: ~22M (only top layers trained)
- Training Time: ~2-3 minutes (with GPU)

## 🔍 Model Explainability

### Grad-CAM (Gradient-weighted Class Activation Mapping)
- Visualizes which regions of the image the model focuses on
- Helps understand model decision-making process
- Useful for debugging and validation

### LIME (Local Interpretable Model-agnostic Explanations)
- Provides local explanations for individual predictions
- Highlights important features for each prediction
- Model-agnostic approach

## 🛠️ Key Technologies

- **TensorFlow/Keras**: Deep learning framework
- **tf.data API**: Efficient input pipelines
- **Data Augmentation**: RandomFlip, RandomZoom, RandomContrast
- **Transfer Learning**: Xception pre-trained on ImageNet
- **Callbacks**: ReduceLROnPlateau, Custom accuracy callback
- **Visualization**: Matplotlib, Seaborn
- **Explainability**: Grad-CAM, LIME

## 📝 Configuration Parameters

```python
batch_size = 64          # Batch size for training
epochs = 200             # Maximum training epochs
img_height = 299         # Image height
img_width = 299          # Image width
seed = 0                 # Random seed for reproducibility
```

## 🎓 Educational Use

This project is designed as a comprehensive lab exercise for students learning:
- Deep learning fundamentals
- Computer vision applications
- Industrial quality control
- Model interpretability
- Best practices in ML development

## 🔧 Troubleshooting

### Out of Memory Error
- Reduce `batch_size` (e.g., from 64 to 32 or 16)
- Reduce image size (e.g., from 299 to 224)
- Use a smaller model (e.g., MobileNet instead of Xception)

### Slow Training
- Enable GPU acceleration
- Reduce number of epochs
- Use transfer learning instead of training from scratch
- Optimize data pipeline with `prefetch()` and `cache()`

### Low Accuracy
- Increase training epochs
- Adjust learning rate
- Try different data augmentation strategies
- Use a deeper network or pre-trained model
- Check data quality and balance

## 📚 References

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [Xception Paper](https://arxiv.org/abs/1610.02357)

## 📄 License

This project is for educational purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

**Happy Learning! 🎉**

