"""
Pill Image Classifier Training Script for Project Aegis

Trains a MobileNetV2-based pill classifier that can be exported to TensorFlow.js.

Usage:
    python train_pill_model.py --data-dir ./pill_data --epochs 30

Data directory structure:
    pill_data/
        train/
            aspirin/
                img001.jpg
                img002.jpg
            ibuprofen/
                ...
        val/
            aspirin/
                ...

After training, convert to TF.js:
    tensorflowjs_converter --input_format=tf_saved_model ./saved_model ./tfjs_model
    
Then place the output in public/models/pill-classifier/

@author Project Aegis
"""

import os
import argparse
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(data_dir, output_dir, epochs=30, batch_size=32, learning_rate=1e-3):
    """Train a MobileNetV2-based pill classifier."""
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        logger.error(
            "TensorFlow not installed. Install with: pip install tensorflow"
        )
        return

    IMG_SIZE = (224, 224)

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=360,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        logger.error(f"Training directory not found: {train_dir}")
        return

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
    )

    val_gen = None
    if os.path.isdir(val_dir):
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=IMG_SIZE,
            batch_size=batch_size,
            class_mode="categorical",
        )

    num_classes = len(train_gen.class_indices)
    logger.info(f"Found {num_classes} classes: {list(train_gen.class_indices.keys())}")

    # Build model: MobileNetV2 base + custom head
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # Freeze base initially

    model = models.Sequential(
        [
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    # Phase 1: Train head only
    logger.info("Phase 1: Training classifier head...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
    ]

    head_epochs = min(epochs // 2, 15)
    model.fit(
        train_gen,
        epochs=head_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Phase 2: Fine-tune top layers of MobileNet
    logger.info("Phase 2: Fine-tuning top layers...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_gen,
        epochs=epochs - head_epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    saved_model_path = os.path.join(output_dir, "saved_model")
    model.save(saved_model_path)
    logger.info(f"Model saved to {saved_model_path}")

    # Save labels
    labels = {v: k for k, v in train_gen.class_indices.items()}
    labels_list = [labels[i] for i in range(num_classes)]
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels_list, f, indent=2)
    logger.info(f"Labels saved to {labels_path}")

    # Convert to TF.js
    try:
        import tensorflowjs as tfjs

        tfjs_path = os.path.join(output_dir, "tfjs_model")
        tfjs.converters.save_keras_model(model, tfjs_path)
        logger.info(f"TF.js model saved to {tfjs_path}")
        logger.info(
            f"Copy {tfjs_path}/* to public/models/pill-classifier/ in your frontend"
        )
    except ImportError:
        logger.warning(
            "tensorflowjs not installed. Convert manually with:\n"
            f"  tensorflowjs_converter --input_format=tf_saved_model {saved_model_path} {output_dir}/tfjs_model"
        )

    if val_gen:
        loss, acc = model.evaluate(val_gen)
        logger.info(f"Validation Loss: {loss:.4f}, Accuracy: {acc:.4f}")


def download_pill_data(output_dir):
    """
    Download pill image data from NIH Pill Image Recognition Challenge.
    This is a placeholder - actual data requires NIH registration.
    """
    logger.info("Pill image datasets available from:")
    logger.info("  - NIH Pill Image Recognition Challenge")
    logger.info("  - FDA Pill Image Database")
    logger.info("  - Custom collection via pharmacy partnerships")
    logger.info("")
    logger.info(f"Place images in {output_dir}/train/<drug_name>/ and {output_dir}/val/<drug_name>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train pill detection model")
    parser.add_argument("--data-dir", required=True, help="Directory with train/val pill images")
    parser.add_argument("--output-dir", default="./pill_model_output", help="Output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--download-data", action="store_true", help="Show data download instructions")

    args = parser.parse_args()

    if args.download_data:
        download_pill_data(args.data_dir)
    else:
        train_model(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
