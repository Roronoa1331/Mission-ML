"""
Complete Casting Defect Detection Pipeline - FIXED VERSION
Compatible with older TensorFlow versions
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)

# Configuration
class Config:
    DATA_DIR = "./casting_512x512/"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS_CUSTOM = 20
    EPOCHS_TRANSFER = 10
    LEARNING_RATE = 0.001

config = Config()

# Custom metrics for older TF versions
def precision_metric(y_true, y_pred):
    """Custom precision metric"""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_metric(y_true, y_pred):
    """Custom recall metric"""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def explore_dataset():
    """Comprehensive dataset exploration"""
    print("=== Dataset Exploration ===")
    
    if not os.path.exists(config.DATA_DIR):
        print(f"Error: Dataset not found at {config.DATA_DIR}")
        print("Please ensure the dataset structure is: casting_512x512/casting_512x512/ok_front/ and /def_front/")
        return None
    
    # Count images
    def_dir = os.path.join(config.DATA_DIR, "def_front")
    ok_dir = os.path.join(config.DATA_DIR, "ok_front")
    
    def_count = len(os.listdir(def_dir)) if os.path.exists(def_dir) else 0
    ok_count = len(os.listdir(ok_dir)) if os.path.exists(ok_dir) else 0
    
    print(f"Defective images: {def_count}")
    print(f"OK images: {ok_count}")
    print(f"Total images: {def_count + ok_count}")
    
    if def_count + ok_count == 0:
        print("Error: No images found in dataset directories")
        return None
        
    print(f"Defect rate: {def_count/(def_count+ok_count)*100:.1f}%")
    
    # Display sample images
    display_sample_images(def_dir, ok_dir)
    
    return config.DATA_DIR

def display_sample_images(def_dir, ok_dir):
    """Display sample images from both classes"""
    plt.figure(figsize=(15, 6))
    
    # Defective samples
    def_files = os.listdir(def_dir)[:4]
    for i, file in enumerate(def_files):
        img_path = os.path.join(def_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 4, i + 1)
            plt.imshow(img)
            plt.title(f"Defective Sample {i+1}")
            plt.axis('off')
    
    # OK samples
    ok_files = os.listdir(ok_dir)[:4]
    for i, file in enumerate(ok_files):
        img_path = os.path.join(ok_dir, file)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 4, i + 5)
            plt.imshow(img)
            plt.title(f"OK Sample {i+1}")
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_generators(data_dir):
    """Create data generators with augmentation"""
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    
    # Validation data (only rescaling)
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(config.IMG_HEIGHT, config.IMG_WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, validation_generator

def create_custom_cnn_model():
    """Create custom CNN model - SIMPLIFIED VERSION"""
    model = keras.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                      input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third conv block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model with only accuracy metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']  # Only using accuracy to avoid metric issues
    )
    
    return model

def create_transfer_learning_model():
    """Create transfer learning model using Xception"""
    # Load pre-trained Xception model
    base_model = Xception(
        weights='imagenet',
        include_top=False,
        input_shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3)
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classifier
    inputs = keras.Input(shape=(config.IMG_HEIGHT, config.IMG_WIDTH, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    # Compile model with only accuracy metric
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE/10),
        loss='binary_crossentropy',
        metrics=['accuracy']  # Only using accuracy to avoid metric issues
    )
    
    return model, base_model

def plot_training_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(15, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_training_history.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def evaluate_model(model, generator, model_name):
    """Comprehensive model evaluation using sklearn"""
    print(f"\n=== {model_name} Evaluation ===")
    
    # Get predictions
    y_true = generator.classes
    y_pred_proba = model.predict(generator)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate metrics using sklearn
    accuracy = np.mean(y_true == y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['OK', 'Defective'],
                yticklabels=['OK', 'Defective'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['OK', 'Defective']))
    
    return accuracy, precision, recall, f1

def main():
    """Main training pipeline"""
    print("=== Casting Defect Detection - Complete Pipeline ===")
    
    # Step 1: Data exploration
    data_dir = explore_dataset()
    if data_dir is None:
        print("Please download the dataset and ensure correct directory structure.")
        print("Dataset should be at: ./casting_512x512/casting_512x512/")
        return
    
    # Step 2: Create data generators
    print("\n=== Creating Data Generators ===")
    train_generator, validation_generator = create_data_generators(data_dir)
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    
    # Step 3: Train custom CNN model
    print("\n=== Training Custom CNN Model ===")
    custom_model = create_custom_cnn_model()
    custom_model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]
    
    # Train custom model
    print("Starting training... This may take a few minutes.")
    custom_history = custom_model.fit(
        train_generator,
        epochs=config.EPOCHS_CUSTOM,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(custom_history, "Custom CNN")
    
    # Evaluate custom model
    custom_accuracy, custom_precision, custom_recall, custom_f1 = evaluate_model(
        custom_model, validation_generator, "Custom CNN"
    )
    
    # Save custom model
    custom_model.save('custom_cnn_model.h5')
    print("Custom CNN model saved as 'custom_cnn_model.h5'")
    
    # Step 4: Train transfer learning model
    print("\n=== Training Transfer Learning Model ===")
    transfer_model, base_model = create_transfer_learning_model()
    transfer_model.summary()
    
    # Train transfer model
    print("Starting transfer learning training...")
    transfer_history = transfer_model.fit(
        train_generator,
        epochs=config.EPOCHS_TRANSFER,
        validation_data=validation_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(transfer_history, "Transfer Learning")
    
    # Evaluate transfer model
    transfer_accuracy, transfer_precision, transfer_recall, transfer_f1 = evaluate_model(
        transfer_model, validation_generator, "Transfer Learning"
    )
    
    # Save transfer model
    transfer_model.save('xception_transfer_model.h5')
    print("Transfer learning model saved as 'xception_transfer_model.h5'")
    
    # Step 5: Model comparison
    print("\n=== Model Comparison ===")
    comparison_data = {
        'Model': ['Custom CNN', 'Transfer Learning'],
        'Accuracy': [custom_accuracy, transfer_accuracy],
        'Precision': [custom_precision, transfer_precision],
        'Recall': [custom_recall, transfer_recall],
        'F1-Score': [custom_f1, transfer_f1]
    }
    
    # Plot comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, [custom_accuracy, custom_precision, custom_recall, custom_f1], 
            width, label='Custom CNN')
    plt.bar(x + width/2, [transfer_accuracy, transfer_precision, transfer_recall, transfer_f1], 
            width, label='Transfer Learning')
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*50)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*50)
    print("\nGenerated files:")
    print("📊 Visualizations:")
    print("   - dataset_samples.png: Sample images from dataset")
    print("   - custom_cnn_training_history.png: Custom CNN training curves")
    print("   - transfer_learning_training_history.png: Transfer learning training curves")
    print("   - model_comparison.png: Model performance comparison")
    print("   - *_confusion_matrix.png: Confusion matrices")
    print("\n🤖 Models:")
    print("   - custom_cnn_model.h5: Trained custom CNN model")
    print("   - xception_transfer_model.h5: Trained transfer learning model")
    print(f"\n📈 Best Model: {'Transfer Learning' if transfer_accuracy > custom_accuracy else 'Custom CNN'}")
    print(f"   Best Accuracy: {max(custom_accuracy, transfer_accuracy):.3f}")

if __name__ == "__main__":
    main()