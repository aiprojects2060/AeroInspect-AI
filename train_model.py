"""
VGG16-based Aircraft Damage Classification Model
Tasks 1-7 from the final project
"""
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.applications import VGG16
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Reproducibility ──────────────────────────────────────────────────────────
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# ── Config ────────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
N_EPOCHS = 5
IMG_ROWS, IMG_COLS = 224, 224
INPUT_SHAPE = (IMG_ROWS, IMG_COLS, 3)


def build_generators(train_dir, valid_dir, test_dir):
    """
    Tasks 1 & 2 – Build data generators.
    Returns (train_generator, valid_generator, test_generator)
    """
    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen  = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_ROWS, IMG_COLS),
        batch_size=BATCH_SIZE,
        seed=seed_value,
        class_mode='binary',
        shuffle=True,
    )

    # Task 1
    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        class_mode='binary',
        seed=seed_value,
        batch_size=BATCH_SIZE,
        shuffle=False,
        target_size=(IMG_ROWS, IMG_COLS),
    )

    # Task 2
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        class_mode='binary',
        seed=seed_value,
        batch_size=BATCH_SIZE,
        shuffle=False,
        target_size=(IMG_ROWS, IMG_COLS),
    )

    return train_generator, valid_generator, test_generator


def build_model():
    """
    Tasks 3 & 4 – Build and compile the VGG16-based model.
    """
    # Task 3 – Load VGG16
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=INPUT_SHAPE,
    )

    output = base_model.layers[-1].output
    output = keras.layers.Flatten()(output)
    base_model = Model(base_model.input, output)

    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    # Task 4 – Compile
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    return model


def train_model(model, train_generator, valid_generator, n_epochs=N_EPOCHS):
    """
    Task 5 – Train the model.
    Returns the history object.
    """
    history = model.fit(
        train_generator,
        epochs=n_epochs,
        validation_data=valid_generator,
    )
    return history


def plot_loss_curves(train_history, save_dir=None):
    """Plot training and validation loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#0e1117')

    for ax in axes:
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='#cdd6f4')
        ax.spines[:].set_color('#313244')
        ax.yaxis.label.set_color('#cdd6f4')
        ax.xaxis.label.set_color('#cdd6f4')
        ax.title.set_color('#cdd6f4')

    axes[0].plot(train_history['loss'], color='#89b4fa', linewidth=2, label='Train Loss')
    axes[0].set_title("Training Loss", fontsize=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(facecolor='#313244', labelcolor='white')

    axes[1].plot(train_history['val_loss'], color='#fab387', linewidth=2, label='Val Loss')
    axes[1].set_title("Validation Loss", fontsize=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend(facecolor='#313244', labelcolor='white')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "loss_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close()
        return path
    return fig


def plot_accuracy_curves(train_history, save_dir=None):
    """Task 6 – Plot accuracy curves for training and validation sets."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1d2e')
    ax.tick_params(colors='#cdd6f4')
    ax.spines[:].set_color('#313244')
    ax.yaxis.label.set_color('#cdd6f4')
    ax.xaxis.label.set_color('#cdd6f4')
    ax.title.set_color('#cdd6f4')

    ax.plot(train_history['accuracy'], color='#a6e3a1', linewidth=2, label='Training Accuracy')
    ax.plot(train_history['val_accuracy'], color='#f38ba8', linewidth=2, label='Validation Accuracy')
    ax.set_title('Accuracy Curve', fontsize=15)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend(facecolor='#313244', labelcolor='white')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "accuracy_curves.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close()
        return path
    return fig


def evaluate_model(model, test_generator):
    """Evaluate model and return (test_loss, test_accuracy)."""
    steps = test_generator.samples // test_generator.batch_size
    test_loss, test_accuracy = model.evaluate(test_generator, steps=steps)
    return test_loss, test_accuracy


def get_predictions(test_generator, model):
    """Task 7 helper – get all predictions."""
    test_generator.reset()
    steps = test_generator.samples // test_generator.batch_size + 1
    preds = model.predict(test_generator, steps=steps)
    predicted_classes = (preds > 0.5).astype(int).flatten()
    true_classes = test_generator.classes[: len(predicted_classes)]
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}
    return predicted_classes, true_classes, class_names


def plot_prediction_grid(test_generator, model, num_images=9, save_dir=None):
    """Task 7 – Show a grid of test images with true/predicted labels."""
    test_generator.reset()
    images, labels = next(test_generator)
    preds = model.predict(images)
    predicted_classes = (preds > 0.5).astype(int).flatten()
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    n = min(num_images, len(images))
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    fig.patch.set_facecolor('#0e1117')
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis('off')
        true_name = class_names[int(labels[i])]
        pred_name = class_names[predicted_classes[i]]
        color = '#a6e3a1' if true_name == pred_name else '#f38ba8'
        ax.set_title(
            f"True: {true_name}\nPred: {pred_name}",
            color=color, fontsize=10, fontweight='bold'
        )

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_dir:
        path = os.path.join(save_dir, "predictions_grid.png")
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#0e1117')
        plt.close()
        return path
    return fig


def save_model(model, save_path):
    model.save(save_path)


def load_saved_model(save_path):
    return keras.models.load_model(save_path)
