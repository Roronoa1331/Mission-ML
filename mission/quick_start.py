"""
Quick Start Script for Casting Defect Detection
Simplified version for quick experiments and learning
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print("TensorFlow version:", tf.__version__)

# Configuration
DATA_DIR = "./casting_512x512/"  # Adjust path as needed
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 10

def load_and_explore_data():
    """Load and explore the dataset"""
    print("Loading dataset...")
    
    # Check if dataset exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory not found at {DATA_DIR}")
        print("Please ensure the dataset is downloaded and extracted correctly")
        return None
    
    # Count images
    def_count = len(os.listdir(os.path.join(DATA_DIR, "def_front")))
    ok_count = len(os.listdir(os.path.join(DATA_DIR, "ok_front")))
    
    print(f"Defective images: {def_count}")
    print(f"OK images: {ok_count}")
    print(f"Total images: {def_count + ok_count}")
    
    return DATA_DIR

def create_data_generators(data_dir):
    """Create data generators with augmentation"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Only rescaling for validation
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    
    # Train generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    
    # Validation generator
    validation_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    
    return train_generator, validation_generator

def create_simple_model():
    """Create a simple CNN model"""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_sample_images(generator):
    """Plot sample images from generator"""
    plt.figure(figsize=(12, 8))
    
    # Get a batch of images
    images, labels = next(generator)
    
    for i in range(8):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {int(labels[i])} (0=OK, 1=Defect)")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_start_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function"""
    print("=== Casting Defect Detection - Quick Start ===")
    
    # Step 1: Load data
    data_dir = load_and_explore_data()
    if data_dir is None:
        return
    
    # Step 2: Create data generators
    print("\nCreating data generators...")
    train_generator, validation_generator = create_data_generators(data_dir)
    
    # Step 3: Plot sample images
    print("Plotting sample images...")
    plot_sample_images(train_generator)
    
    # Step 4: Create and train model
    print("\nCreating model...")
    model = create_simple_model()
    model.summary()
    
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Step 5: Evaluate model
    print("\nEvaluating model...")
    train_loss, train_accuracy = model.evaluate(train_generator)
    val_loss, val_accuracy = model.evaluate(validation_generator)
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Step 6: Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('quick_start_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Step 7: Save model
    model.save('quick_start_model.h5')
    print("\nModel saved as 'quick_start_model.h5'")
    
    # Step 8: Make some predictions
    print("\nMaking predictions on validation set...")
    images, true_labels = next(validation_generator)
    predictions = model.predict(images)
    predicted_labels = (predictions > 0.5).astype(int).flatten()
    
    # Plot predictions
    plt.figure(figsize=(15, 10))
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.imshow(images[i])
        color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
        plt.title(f"True: {int(true_labels[i])}, Pred: {predicted_labels[i]}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('quick_start_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n=== Quick Start Completed ===")
    print("Generated files:")
    print("- quick_start_samples.png: Sample images from dataset")
    print("- quick_start_training_history.png: Training curves")
    print("- quick_start_predictions.png: Prediction results")
    print("- quick_start_model.h5: Trained model")

if __name__ == "__main__":
    main()