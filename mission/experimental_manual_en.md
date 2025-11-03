# Industrial Casting Defect Detection Experimental Manual

## Experiment Overview

### Experiment Objectives
This experiment aims to enable students to master the complete process of using deep learning for industrial quality control, including:
1. Image data preprocessing and augmentation
2. Building custom CNN models
3. Using transfer learning to improve model performance
4. Model interpretability analysis (Grad-CAM and LIME)

### Experiment Background
In industrial production, product quality inspection is a critical component. This experiment uses a real industrial casting dataset to automatically identify surface defects in castings through deep learning models, achieving intelligent quality inspection.

### Dataset Description
- **Data Source**: Real industrial casting product dataset
- **Image Size**: 512x512 pixels
- **Classes**: 
  - `ok_front`: Qualified products
  - `def_front`: Defective products
- **Total Samples**: Approximately 1300 images

### Experiment Environment
- Python 3.7+
- TensorFlow 2.10+
- GPU (recommended for acceleration)

---

## Experiment Preparation

### Step 1: Environment Configuration

#### 1.1 Create Virtual Environment (Recommended)
```bash
# Create virtual environment with conda
conda create -n casting_detection python=3.9
conda activate casting_detection

# Or use venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# Or
venv\Scripts\activate  # Windows
```

#### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

#### 1.3 Verify Installation
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
```

### Step 2: Data Preparation

#### 2.1 Download Dataset
Download dataset from Kaggle:
- Dataset name: "Real-life Industrial Dataset of Casting Product"
- Link: https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product

#### 2.2 Organize Data Directory
Extract dataset to project directory, ensure directory structure as follows:
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
├── requirements.txt
└── experimental_manual.md
```

#### 2.3 Modify Configuration
Open `casting_defect_detection.py`, modify dataset path:
```python
dataset_url = "./casting_512x512/"  # Modify according to actual path
```

---

## Experiment 1: Data Exploration and Preprocessing

### Experiment Objectives
- Understand dataset structure and characteristics
- Master image data loading and visualization methods
- Learn data preprocessing techniques

### Experiment Steps

#### Step 1: Data Exploration
Create file `experiment_1_data_exploration.py`:

```python
import pathlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
import PIL

# Set data path
data_dir = pathlib.Path("./casting_512x512/")

# Count images
image_count = len(list(data_dir.glob('*/*.jpeg')))
print(f"Total dataset images: {image_count}")

# View defect samples
def_front = list(data_dir.glob('def_front/*'))
print(f"Defect samples: {len(def_front)}")

# View qualified samples
ok_front = list(data_dir.glob('ok_front/*'))
print(f"Qualified samples: {len(ok_front)}")

# Visualize samples
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Show defect samples
for i in range(3):
    img = PIL.Image.open(def_front[i])
    axes[0, i].imshow(img)
    axes[0, i].set_title(f"Defect Sample {i+1}")
    axes[0, i].axis('off')

# Show qualified samples
for i in range(3):
    img = PIL.Image.open(ok_front[i])
    axes[1, i].imshow(img)
    axes[1, i].set_title(f"Qualified Sample {i+1}")
    axes[1, i].axis('off')

plt.tight_layout()
plt.savefig('data_samples.png')
plt.show()

# Check image dimensions
sample = imread(ok_front[0])
print(f"\nImage shape: {sample.shape}")
print(f"Image data type: {sample.dtype}")
```

Run script:
```bash
python experiment_1_data_exploration.py
```

#### Step 2: Dataset Splitting
Create file `experiment_2_data_preparation.py`:

```python
import tensorflow as tf

# Configuration parameters
batch_size = 64
img_height = 299
img_width = 299
seed = 0

data_dir = "./casting_512x512/"

# Create training set (80%)
train_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    class_names=['ok_front', 'def_front'],
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Create validation set (20%)
val_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    class_names=['ok_front', 'def_front'],
    subset="validation",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_set.class_names
print(f"Class names: {class_names}")

# Visualize batch data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_set.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.savefig('batch_samples.png')
plt.show()

# Check data shapes
for images, labels in train_set:
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    break
```

Run script:
```bash
python experiment_2_data_preparation.py
```

### Discussion Questions
1. Is the dataset balanced? If not, how should it be handled?
2. Why choose 299x299 image size?
3. What impact does batch size have on training?

---

## Experiment 2: Data Augmentation

### Experiment Objectives
- Understand the role and principles of data augmentation
- Master TensorFlow/Keras data augmentation techniques
- Observe effects of different augmentation strategies

### Experiment Steps

#### Step 1: Implement Data Augmentation
Create file `experiment_3_data_augmentation.py`:

```python
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

# Configuration
img_height = 299
img_width = 299
seed = 0

# Create data augmentation layer
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", 
                     input_shape=(img_height, img_width, 3), 
                     seed=seed),
    layers.RandomZoom(0.1, seed=seed),
    layers.RandomContrast(0.3, seed=seed)
])

# Load a sample image
data_dir = "./casting_512x512/"
train_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=1
)

# Get one image
for images, labels in train_set.take(1):
    original_image = images[0]

# Apply data augmentation and visualize
plt.figure(figsize=(15, 10))

# Show original image
plt.subplot(3, 4, 1)
plt.imshow(original_image.numpy().astype("uint8"))
plt.title("Original Image")
plt.axis('off')

# Show 11 augmented images
for i in range(11):
    augmented_image = data_augmentation(tf.expand_dims(original_image, 0))
    plt.subplot(3, 4, i + 2)
    plt.imshow(augmented_image[0].numpy().astype("uint8"))
    plt.title(f"Augmented {i+1}")
    plt.axis('off')

plt.tight_layout()
plt.savefig('augmentation_examples.png')
plt.show()
```

Run script:
```bash
python experiment_3_data_augmentation.py
```

#### Step 2: Compare Different Augmentation Strategies
Modify code to try the following augmentation strategies:

**Strategy A: Only Flipping**
```python
data_augmentation_a = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed)
])
```

**Strategy B: Flip + Rotation**
```python
data_augmentation_b = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed),
    layers.RandomRotation(0.2, seed=seed)
])
```

**Strategy C: Complete Augmentation**
```python
data_augmentation_c = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed),
    layers.RandomZoom(0.1, seed=seed),
    layers.RandomContrast(0.3, seed=seed)
])
```

### Discussion Questions
1. Why is data augmentation necessary?
2. Which augmentation methods are suitable for this dataset? Why?
3. What problems can excessive data augmentation cause?

---

## Experiment 3: Building and Training Custom CNN Model

### Experiment Objectives
- Master CNN model construction methods
- Understand the role of convolutional, pooling, and fully connected layers
- Learn model training and optimization techniques

### Experiment Steps

#### Step 1: Build Simple CNN Model
Create file `experiment_4_custom_cnn.py`:

```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# Configuration parameters
batch_size = 64
epochs = 50
img_height = 299
img_width = 299
seed = 0

# Prepare data
data_dir = "./casting_512x512/"

train_set = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    seed=seed, image_size=(img_height, img_width), batch_size=batch_size
)

val_set = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    seed=seed, image_size=(img_height, img_width), batch_size=batch_size
)

# Optimize data pipeline
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_set.cache().shuffle(1300).prefetch(buffer_size=AUTOTUNE)
val_ds = val_set.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed),
    layers.RandomZoom(0.1, seed=seed),
    layers.RandomContrast(0.3, seed=seed)
])

# Build model
model = Sequential([
    layers.Rescaling(1./255),
    data_augmentation,
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# View model structure
model.build((None, img_height, img_width, 3))
model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Plot training curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('custom_cnn_training.png')
plt.show()

# Save model
model.save('custom_cnn_model.h5')
print("Model saved!")
```

Run script:
```bash
python experiment_4_custom_cnn.py
```

#### Step 2: Add Callback Functions to Optimize Training
Modify training code to add callbacks:

```python
# Custom callback: stop training when target accuracy is reached
class AccuracyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99 and logs.get('val_accuracy') >= 0.99:
            print("\nReached 99% accuracy, stopping training!")
            self.model.stop_training = True

# Learning rate decay callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1,
    min_delta=0.01,
    min_lr=0.000001
)

# Train with callbacks
callbacks = [AccuracyCallback(), reduce_lr]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)
```

### Experiment Tasks
1. **Task 1**: Modify model structure, try different numbers of convolutional layers and filters
2. **Task 2**: Compare effects of different optimizers (Adam, SGD, RMSprop)
3. **Task 3**: Adjust learning rate, observe impact on training

### Discussion Questions
1. Why use MaxPooling layers?
2. What is the role of Dropout layers? Where is appropriate to add them?
3. How to determine if model is overfitting?

---

## Experiment 4: Transfer Learning

### Experiment Objectives
- Understand principles and advantages of transfer learning
- Master methods of using pre-trained models
- Compare performance between custom models and transfer learning

### Experiment Steps

#### Step 1: Use Xception Pre-trained Model
Create file `experiment_5_transfer_learning.py`:

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Configuration parameters
batch_size = 64
epochs = 50
img_height = 299
img_width = 299
seed = 0

# Prepare data (same as Experiment 3)
data_dir = "./casting_512x512/"

train_set = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="training",
    seed=seed, image_size=(img_height, img_width), batch_size=batch_size
)

val_set = tf.keras.utils.image_dataset_from_directory(
    data_dir, validation_split=0.2, subset="validation",
    seed=seed, image_size=(img_height, img_width), batch_size=batch_size
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_set.cache().shuffle(1300).prefetch(buffer_size=AUTOTUNE)
val_ds = val_set.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical", seed=seed),
    layers.RandomZoom(0.1, seed=seed),
    layers.RandomContrast(0.3, seed=seed)
])

# Load pre-trained Xception model
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(img_height, img_width, 3),
    include_top=False  # Exclude top classifier
)

# Freeze base model weights
base_model.trainable = False

# Build new model
inputs = keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = keras.layers.Rescaling(scale=1 / 255.0)(x)
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# View model structure
model.summary()

# Train model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Plot training curves
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Xception Transfer Learning - Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Xception Transfer Learning - Loss')
plt.savefig('transfer_learning_training.png')
plt.show()

# Save model
model.save('xception_transfer_model.h5')
print("Transfer learning model saved!")
```

Run script:
```bash
python experiment_5_transfer_learning.py
```

#### Step 2: Fine-tuning
After initial model training, unfreeze some layers for fine-tuning:

```python
# Unfreeze top layers of base model
base_model.trainable = True

# Freeze earlier layers, only train later layers
fine_tune_at = 100  # Fine-tune from layer 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with smaller learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Continue training
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20
)

# Save fine-tuned model
model.save('xception_finetuned_model.h5')
```

### Experiment Tasks
1. **Task 1**: Try other pre-trained models (ResNet50, VGG16, MobileNet)
2. **Task 2**: Compare effects of freezing different numbers of layers
3. **Task 3**: Compare training time and accuracy between transfer learning and custom models

### Discussion Questions
1. Why can transfer learning accelerate training?
2. When should transfer learning be used?
3. Why use smaller learning rate during fine-tuning?

---

## Experiment 5: Model Evaluation and Optimization

### Experiment Objectives
- Master multiple model evaluation metrics
- Learn confusion matrix usage
- Understand model optimization strategies

### Experiment Steps

#### Step 1: Comprehensive Model Evaluation
Create file `experiment_6_model_evaluation.py`:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Load model
model = keras.models.load_model('xception_transfer_model.h5')

# Prepare test data
data_dir = "./casting_512x512/"
img_height = 299
img_width = 299
batch_size = 32

val_set = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=0,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

# Get true labels and predictions
y_true = []
y_pred = []

for images, labels in val_set:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend((predictions > 0.5).astype(int).flatten())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('confusion_matrix.png')
plt.show()

# Print classification report
class_names = ['ok_front', 'def_front']
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Calculate various metrics
tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
```

Run script:
```bash
python experiment_6_model_evaluation.py
```

#### Step 2: Error Analysis
Analyze incorrectly predicted samples:

```python
# Find incorrectly predicted samples
wrong_indices = np.where(y_true != y_pred)[0]
print(f"\nNumber of incorrectly predicted samples: {len(wrong_indices)}")

# Visualize incorrect samples
if len(wrong_indices) > 0:
    plt.figure(figsize=(15, 10))
    count = 0

    for images, labels in val_set:
        for i in range(len(images)):
            if count in wrong_indices:
                predictions = model.predict(images[i:i+1])
                pred_class = int(predictions[0] > 0.5)
                true_class = int(labels[i])

                idx = wrong_indices.tolist().index(count)
                if idx < 9:
                    plt.subplot(3, 3, idx + 1)
                    plt.imshow(images[i].numpy().astype("uint8"))
                    plt.title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}")
                    plt.axis('off')

            count += 1
            if count > max(wrong_indices):
                break
        if count > max(wrong_indices):
            break

    plt.tight_layout()
    plt.savefig('wrong_predictions.png')
    plt.show()
```

### Experiment Tasks
1. **Task 1**: Analyze which type of error is more common (false positives or false negatives)
2. **Task 2**: Observe common characteristics of incorrect samples
3. **Task 3**: Propose suggestions for model improvement

### Discussion Questions
1. In quality inspection scenarios, which is more serious: false positives or false negatives?
2. How to handle class imbalance problems?
3. Besides accuracy, what other metrics should be considered?

---

## Experiment 6: Model Interpretability Analysis

### Experiment Objectives
- Understand the importance of model decision interpretability
- Master Grad-CAM visualization techniques
- Learn to use LIME for local explanations

### Experiment Steps

#### Step 1: Grad-CAM Visualization
Create file `experiment_7_gradcam.py`:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

def get_img_array(img_path, size):
    """Load and preprocess image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Generate Grad-CAM heatmap"""
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, alpha=0.4):
    """Save and display Grad-CAM visualization"""
    img = keras.preprocessing.image.load_img(img_path)
    img = keras.preprocessing.image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap)

    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)

    # Visualization
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(Image.open(img_path))
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    plt.title("Grad-CAM Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png')
    plt.show()

# Main program
if __name__ == "__main__":
    # Load model
    model = keras.models.load_model('xception_transfer_model.h5')

    # Select a test image
    img_path = "./casting_512x512/def_front/cast_def_0_1.jpeg"  # Modify to actual path
    img_size = (299, 299)

    # Preprocess image
    preprocess_input = keras.applications.xception.preprocess_input
    img_array = preprocess_input(get_img_array(img_path, size=img_size))

    # Get prediction
    preds = model.predict(img_array)
    print(f"Prediction: {'Defective' if preds[0] < 0.5 else 'Qualified'}")
    print(f"Confidence: {abs(preds[0][0] - 0.5) * 2:.2%}")

    # Generate Grad-CAM
    # Note: Need to find the name of the last convolutional layer in Xception model
    last_conv_layer_name = "block14_sepconv2_act"

    # Create Grad-CAM model
    grad_cam_model = keras.applications.Xception(weights="imagenet")
    grad_cam_model.layers[-1].activation = None

    heatmap = make_gradcam_heatmap(img_array, grad_cam_model, last_conv_layer_name)

    # Display results
    save_and_display_gradcam(img_path, heatmap)
```

Run script:
```bash
python experiment_7_gradcam.py
```

#### Step 2: LIME Interpretability Analysis
Create file `experiment_8_lime.py`:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

def get_img_array(img_path, size):
    """Load and preprocess image"""
    img = keras.preprocessing.image.load_img(img_path, target_size=size)
    array = keras.preprocessing.image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array

# Load model
model = keras.models.load_model('custom_cnn_model.h5')

# Select test image
img_path = "./casting_512x512/def_front/cast_def_0_1.jpeg"
img_size = (299, 299)

# Load image
images = get_img_array(img_path, img_size)

# Get prediction
preds = model.predict(images)
prediction = np.argmax(preds)
confidence = np.max(preds)

print(f"Predicted class: {'Defective' if prediction == 0 else 'Qualified'}")
print(f"Confidence: {confidence:.4f}")

# Create LIME explainer
explainer = lime_image.LimeImageExplainer()

# Generate explanation
print("\nGenerating LIME explanation (this may take several minutes)...")
explanation = explainer.explain_instance(
    images[0].astype('double'),
    model.predict,
    top_labels=2,
    hide_color=0,
    num_samples=300
)

# Visualize explanation
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=False
)

# Display results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(images[0].astype('uint8'))
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mark_boundaries(temp, mask))
plt.title("LIME Explanation - Important Regions")
plt.axis('off')

# Show only important regions
temp_pos, mask_pos = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=True
)

plt.subplot(1, 3, 3)
plt.imshow(temp_pos)
plt.title("LIME Explanation - Only Important Regions")
plt.axis('off')

plt.tight_layout()
plt.savefig('lime_explanation.png')
plt.show()

print("\nLIME explanation generated!")
```

Run script:
```bash
python experiment_8_lime.py
```

### Experiment Tasks
1. **Task 1**: Compare Grad-CAM results of multiple images, analyze regions model focuses on
2. **Task 2**: Use LIME to analyze correctly and incorrectly predicted samples
3. **Task 3**: Summarize the basis for model defect judgment

### Discussion Questions
1. What are the differences between Grad-CAM and LIME?
2. Do the regions model focuses on align with human judgment?
3. How to use interpretability analysis to improve the model?

---

## Experiment 7: Model Deployment and Application

### Experiment Objectives
- Learn model saving and loading
- Master batch prediction methods
- Understand basic model deployment process

### Experiment Steps

#### Step 1: Create Prediction Script
Create file `predict.py`:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def load_and_predict(model_path, image_path, img_size=(299, 299)):
    """Load model and make prediction"""
    # Load model
    model = keras.models.load_model(model_path)

    # Load image
    img = keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    # Parse result
    class_names = ['ok_front', 'def_front']
    pred_class = int(prediction[0] > 0.5)
    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]

    return class_names[pred_class], confidence

def batch_predict(model_path, image_dir, img_size=(299, 299)):
    """Batch prediction"""
    model = keras.models.load_model(model_path)
    image_dir = Path(image_dir)

    results = []

    for img_path in image_dir.glob('*.jpeg'):
        img = keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        pred_class = int(prediction[0] > 0.5)
        confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]

        results.append({
            'image': img_path.name,
            'prediction': 'ok_front' if pred_class == 1 else 'def_front',
            'confidence': confidence
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Casting Defect Detection Prediction')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--batch', type=str, help='Batch prediction directory')

    args = parser.parse_args()

    if args.image:
        # Single prediction
        pred_class, confidence = load_and_predict(args.model, args.image)
        print(f"\nPrediction result: {pred_class}")
        print(f"Confidence: {confidence:.2%}")

        # Display image
        img = plt.imread(args.image)
        plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.title(f"Prediction: {pred_class} (Confidence: {confidence:.2%})")
        plt.axis('off')
        plt.show()

    elif args.batch:
        # Batch prediction
        print(f"\nBatch predicting images in {args.batch}...")
        results = batch_predict(args.model, args.batch)

        print(f"\nPredicted {len(results)} images:")
        print("-" * 60)
        for r in results:
            print(f"{r['image']:30s} | {r['prediction']:10s} | {r['confidence']:.2%}")

        # Statistics
        ok_count = sum(1 for r in results if r['prediction'] == 'ok_front')
        def_count = len(results) - ok_count
        print("-" * 60)
        print(f"Qualified: {ok_count} | Defective: {def_count}")
```

Usage:
```bash
# Single image prediction
python predict.py --model xception_transfer_model.h5 --image ./casting_512x512/def_front/cast_def_0_1.jpeg

# Batch prediction
python predict.py --model xception_transfer_model.h5 --batch ./casting_512x512/def_front/
```

#### Step 2: Create Simple Web Application (Optional)
Create file `app.py`:

```python
from flask import Flask, request, render_template, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load model
model = keras.models.load_model('xception_transfer_model.h5')
img_size = (299, 299)

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Casting Defect Detection</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 50px auto; }
            .container { text-align: center; }
            input[type="file"] { margin: 20px 0; }
            #result { margin-top: 20px; font-size: 24px; }
            img { max-width: 500px; margin: 20px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Casting Defect Detection System</h1>
            <input type="file" id="imageInput" accept="image/*">
            <div id="preview"></div>
            <div id="result"></div>
        </div>
        <script>
            document.getElementById('imageInput').addEventListener('change', function(e) {
                const file = e.target.files[0];
                const reader = new FileReader();

                reader.onload = function(event) {
                    const img = new Image();
                    img.src = event.target.result;
                    img.onload = function() {
                        document.getElementById('preview').innerHTML = '<img src="' + event.target.result + '">';

                        // Send to server
                        fetch('/predict', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({image: event.target.result})
                        })
                        .then(response => response.json())
                        .then(data => {
                            const resultDiv = document.getElementById('result');
                            const className = data.prediction === 'ok_front' ? 'Qualified' : 'Defective';
                            const color = data.prediction === 'ok_front' ? 'green' : 'red';
                            resultDiv.innerHTML = `<span style="color: ${color}">Detection Result: ${className}</span><br>Confidence: ${(data.confidence * 100).toFixed(2)}%`;
                        });
                    };
                };
                reader.readAsDataURL(file);
            });
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)

    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize(img_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = int(prediction[0] > 0.5)
    confidence = prediction[0][0] if pred_class == 1 else 1 - prediction[0][0]

    return jsonify({
        'prediction': 'ok_front' if pred_class == 1 else 'def_front',
        'confidence': float(confidence)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

Install Flask and run:
```bash
pip install flask
python app.py
```

Then visit `http://localhost:5000` in browser

### Experiment Tasks
1. **Task 1**: Use prediction script to test multiple images
2. **Task 2**: Analyze prediction error cases
3. **Task 3**: (Optional) Deploy web application and test

---

## Experiment Summary and Report Requirements

### Lab Report Content

#### 1. Experiment Purpose and Principles (10 points)
- Briefly describe deep learning applications in industrial quality inspection
- Explain basic principles of CNN and transfer learning

#### 2. Experiment Process (40 points)
- **Data Exploration**: Dataset statistics, sample visualization
- **Model Building**:
  - Custom CNN model structure and parameters
  - Transfer learning model structure and parameters
- **Training Process**:
  - Training curve graphs
  - Key hyperparameter settings
  - Training time comparison

#### 3. Experiment Results (30 points)
- **Performance Comparison Table**:

| Model | Training Accuracy | Validation Accuracy | Training Time | Parameter Count |
|-------|------------------|-------------------|---------------|-----------------|
| Custom CNN | | | | |
| Xception Transfer Learning | | | | |

- **Confusion Matrix**
- **Classification Report** (precision, recall, F1-score)
- **Error Analysis**: Show incorrectly predicted samples and analyze reasons

#### 4. Interpretability Analysis (10 points)
- Grad-CAM visualization results
- LIME explanation results
- Analyze whether model focuses on reasonable regions

#### 5. Reflection and Summary (10 points)
- Answer discussion questions from each experiment
- Summarize experiment gains
- Propose model improvement suggestions

### Submission Requirements
1. Lab report (PDF format)
2. All experiment code
3. Trained model files
4. Key result charts

---

## FAQ

### Q1: Insufficient GPU memory during training?
**A**:
- Reduce batch_size (e.g., from 64 to 32 or 16)
- Reduce image size (e.g., from 299 to 224)
- Use smaller model (e.g., MobileNet instead of Xception)

### Q2: Training too slow?
**A**:
- Use GPU acceleration
- Reduce epochs number
- Use smaller dataset for experiments
- Use pre-trained models (transfer learning)

### Q3: Low model accuracy?
**A**:
- Increase training epochs
- Adjust learning rate
- Try different data augmentation strategies
- Use deeper networks or pre-trained models
- Check data quality

### Q4: How to choose appropriate pre-trained models?
**A**:
- **Xception**: High accuracy, more parameters, suitable for GPU training
- **MobileNet**: Lightweight, suitable for mobile deployment
- **ResNet50**: Classic model, stable performance
- **VGG16**: Simple structure, easy to understand

### Q5: How to choose data augmentation parameters?
**A**:
- Choose based on actual application scenarios
- In this case, castings are symmetrical, so flipping is appropriate
- Avoid excessive augmentation causing distortion
- Check augmentation effects through visualization

### Q6: How to determine if model is overfitting?
**A**:
- Observe training curves: training accuracy much higher than validation accuracy
- Validation loss starts to increase
- Solutions:
  - Increase data augmentation
  - Add Dropout layers
  - Reduce model complexity
  - Use regularization

### Q7: Grad-CAM display errors?
**A**:
- Check if last convolutional layer name is correct
- Use `model.summary()` to view all layer names
- Ensure using base model rather than complete model

### Q8: LIME running too long?
**A**:
- Reduce `num_samples` parameter (e.g., from 300 to 100)
- Reduce `num_features` parameter
- Use smaller images

---

## Extended Learning

### Advanced Tasks
1. **Multi-class Classification**: Extend to identify multiple defect types
2. **Object Detection**: Use YOLO or Faster R-CNN to locate defect positions
3. **Semantic Segmentation**: Use U-Net to precisely segment defect areas
4. **Model Compression**: Use quantization, pruning to compress model
5. **Edge Deployment**: Deploy model to Raspberry Pi or mobile devices

### Recommended Resources
- **TensorFlow Official Tutorials**: https://www.tensorflow.org/tutorials
- **Keras Documentation**: https://keras.io/
- **Deep Learning Course**: Andrew Ng's Deep Learning Specialization
- **Paper Reading**:
  - "Grad-CAM: Visual Explanations from Deep Networks"
  - "Xception: Deep Learning with Depthwise Separable Convolutions"

---

## Appendix

### A. Complete Project Structure
```
cam/
├── casting_512x512/              # Dataset directory
│   ├── def_front/
│   └── ok_front/
├── casting_defect_detection.py  # Main program
├── requirements.txt              # Dependencies
├── experimental_manual.md        # This manual
├── experiment_1_data_exploration.py
├── experiment_2_data_preparation.py
├── experiment_3_data_augmentation.py
├── experiment_4_custom_cnn.py
├── experiment_5_transfer_learning.py
├── experiment_6_model_evaluation.py
├── experiment_7_gradcam.py
├── experiment_8_lime.py
├── predict.py                    # Prediction script
├── app.py                        # Web application (optional)
├── custom_cnn_model.h5          # Saved model
├── xception_transfer_model.h5   # Saved model
└── results/                      # Results directory
    ├── data_samples.png
    ├── augmentation_examples.png
    ├── custom_cnn_training.png
    ├── transfer_learning_training.png
    ├── confusion_matrix.png
    ├── gradcam_visualization.png
    └── lime_explanation.png
```

### B. Quick Start Guide
```bash
# 1. Clone or download project
cd cam

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset and extract to casting_512x512/

# 4. Run complete training pipeline
python casting_defect_detection.py

# 5. Make predictions
python predict.py --model xception_transfer_model.h5 --image test_image.jpeg
```

---

**Good luck with your experiments! Please contact teaching assistant if you have any questions.**