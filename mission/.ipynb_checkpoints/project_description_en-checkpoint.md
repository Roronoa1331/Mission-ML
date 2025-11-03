# Casting Defect Detection Project - File Description

## 📦 Created Files

### 1. Core Code Files

#### `casting_defect_detection.py` - Main Program
Complete training pipeline including:
- Data exploration and visualization
- Data preprocessing and augmentation
- Custom CNN model construction
- Transfer learning model (Xception)
- Model training and evaluation
- Grad-CAM and LIME interpretability analysis

**Usage:**
```bash
python casting_defect_detection.py
```

#### `quick_start.py` - Quick Start Script
Simplified version suitable for quick experiments and learning:
- Reduced code complexity
- Faster training speed
- Automatic result chart saving
- Beginner-friendly

**Usage:**
```bash
python quick_start.py
```

#### `predict.py` - Prediction Script
For predicting new images:
- Single image prediction support
- Batch prediction support
- Automatic visualization results
- CSV result export capability

**Usage:**
```bash
# Single image
python predict.py --model xception_transfer_model.h5 --image test.jpeg

# Batch prediction
python predict.py --model xception_transfer_model.h5 --batch ./test_images/

# Save results to CSV
python predict.py --model xception_transfer_model.h5 --batch ./test_images/ --output results.csv
```

### 2. Configuration Files

#### `requirements.txt` - Dependency List
Contains all required Python packages:
- TensorFlow 2.10+
- Keras
- NumPy, Pandas
- Matplotlib
- OpenCV, scikit-image
- LIME

**Installation:**
```bash
pip install -r requirements.txt
```

### 3. Documentation Files

#### `实验手册.md` - Complete Experiment Guide (Chinese)
Detailed experiment tutorial including:
- **8 progressive experiments**
- Purpose, steps, and code for each experiment
- Thought questions and experimental tasks
- Common problems FAQ
- Lab report requirements

**Experiment List:**
1. Data exploration and preprocessing
2. Dataset splitting
3. Data augmentation
4. Building and training custom CNN model
5. Transfer learning
6. Model evaluation and optimization
7. Grad-CAM visualization
8. LIME interpretability analysis

#### `README.md` - Project Description (English)
Project overview and quick start guide

#### `项目说明.md` - This File
File structure and usage instructions

## 🎯 Usage Workflow

### Option A: Quick Experience (Recommended for Beginners)

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Prepare Dataset**
   - Download dataset and extract to `./casting_512x512/`
   - Ensure correct directory structure

3. **Run Quick Start Script**
```bash
python quick_start.py
```

4. **View Results**
   - `quick_start_samples.png` - Data samples
   - `quick_start_training_history.png` - Training curves
   - `quick_start_predictions.png` - Prediction results
   - `quick_start_model.h5` - Trained model

### Option B: Complete Experiment (Recommended for Learners)

1. **Complete 8 Experiments Step by Step**
   - Read `实验手册.md`
   - Create separate Python files for each experiment
   - Complete experimental tasks and thought questions

2. **Run Complete Training Pipeline**
```bash
python casting_defect_detection.py
```

3. **Perform Predictions and Analysis**
```bash
python predict.py --model xception_transfer_model.h5 --image test.jpeg
```

### Option C: Direct Usage (Recommended for Practitioners)

1. **Train Model**
```bash
python casting_defect_detection.py
```

2. **Use Model for Predictions**
```bash
python predict.py --model xception_transfer_model.h5 --batch ./new_images/
```

## 📊 Expected Output

### Training Process Output
- Model structure summary
- Training progress per epoch
- Training and validation accuracy, loss
- Learning rate adjustment information

### Generated Files
- `custom_cnn_model.h5` - Custom CNN model
- `xception_transfer_model.h5` - Transfer learning model
- `quick_start_model.h5` - Quick start model
- Various visualization charts (PNG format)

### Model Performance
- **Custom CNN**: Accuracy ~95-100%
- **Transfer Learning**: Accuracy ~98-100%
- **Training Time**: 2-5 minutes (with GPU)

## 🔧 Common Questions

### Q: Where to download the dataset?
A: Download from Kaggle "Real-life Industrial Dataset of Casting Product"
   Link: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

### Q: Can it run without GPU?
A: Yes, but training will be slower. Recommendations:
   - Use `quick_start.py` (faster)
   - Reduce epochs number
   - Use Google Colab (free GPU)

### Q: How to modify training parameters?
A: Modify configuration section at the beginning of scripts:
```python
batch_size = 32  # Reduce to lower memory usage
epochs = 20      # Reduce to speed up training
img_height = 224 # Reduce to speed up training
```

### Q: How to use custom dataset?
A: 
1. Organize data with same directory structure:
```
your_dataset/
├── class1/
│   ├── image1.jpeg
│   └── ...
└── class2/
    ├── image1.jpeg
    └── ...
```
2. Modify `dataset_url` and `class_names` in scripts

### Q: What if prediction results are inaccurate?
A:
1. Increase training epochs
2. Use transfer learning model
3. Adjust data augmentation parameters
4. Check data quality

## 📚 Learning Path

### Beginner (1-2 weeks)
1. Run `quick_start.py` to understand basic workflow
2. Complete experiments 1-3 (data processing)
3. Understand CNN basic concepts

### Intermediate (2-3 weeks)
1. Complete experiments 4-5 (model training)
2. Learn parameter tuning techniques
3. Compare different model performances

### Advanced (3-4 weeks)
1. Complete experiments 6-8 (evaluation and interpretability)
2. Try model optimization
3. Complete comprehensive lab report

## 🎓 Lab Report Template

Lab report should include:

### 1. Experiment Purpose (10%)
- Project background
- Technical principles

### 2. Experiment Process (40%)
- Data exploration
- Model design
- Training process
- Parameter tuning

### 3. Experiment Results (30%)
- Performance metrics
- Comparative analysis
- Visualization charts

### 4. Interpretability Analysis (10%)
- Grad-CAM results
- LIME analysis
- Model decision analysis

### 5. Summary and Thoughts (10%)
- Experiment gains
- Improvement suggestions
- Application prospects

## 📞 Technical Support

When encountering problems:
1. Check FAQ section in `实验手册.md`
2. Check error messages and search for solutions
3. Check TensorFlow official documentation
4. Contact teaching assistant or ask in forum

## 🚀 Advanced Directions

After completing basic experiments, you can try:

1. **Model Optimization**
   - Try other pre-trained models (ResNet, EfficientNet)
   - Implement model ensemble
   - Perform hyperparameter search

2. **Function Extension**
   - Implement multi-class classification
   - Add object detection functionality
   - Develop Web interface

3. **Deployment Application**
   - Convert to TensorFlow Lite
   - Deploy to mobile devices
   - Create REST API

4. **Research Directions**
   - Research few-shot learning
   - Explore active learning
   - Implement online learning

## 📝 Update Log

### Version 1.0
- Initial version
- Complete training and prediction pipeline
- 8 experiment tutorials
- Chinese and English documentation

---

**Happy learning! Contact us anytime with questions.** 🎉