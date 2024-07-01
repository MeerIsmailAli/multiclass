import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Define function to load data with heavy augmentation
def load_data(base_directory, img_height, img_width, batch_size):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.3,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(base_directory, 'train'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse'
    )

    validation_generator = test_datagen.flow_from_directory(
        os.path.join(base_directory, 'validation'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(base_directory, 'test'),
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse'
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    ).repeat()

    validation_dataset = tf.data.Dataset.from_generator(
        lambda: validation_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    test_dataset = tf.data.Dataset.from_generator(
        lambda: test_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, img_height, img_width, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32)
        )
    )

    return train_dataset, validation_dataset, test_dataset, train_generator.samples, validation_generator.samples, test_generator.samples, train_generator.class_indices

# Load data
data_dir = "/home/meer/Desktop/multiclass/multiclass/dataset"
img_height, img_width, batch_size = 50, 50, 2
train_dataset, validation_dataset, test_dataset, train_samples, val_samples, test_samples, class_indices = load_data(data_dir, img_height, img_width, batch_size)

print("Class indices:", class_indices)


# Debug prints to ensure data generators are working
print(f"Number of training samples: {train_samples}")
print(f"Number of validation samples: {val_samples}")

# Check a single batch
x_batch, y_batch = next(iter(train_dataset))
print("Shape of x_batch:", x_batch.shape)
print("Shape of y_batch:", y_batch.shape)

# Load a pretrained model (VGG16) and add custom layers
base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Add custom layers
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Calculate steps per epoch and validation steps
steps_per_epoch = max(1, train_samples // batch_size)
validation_steps = max(1, val_samples // batch_size)

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Validation steps: {validation_steps}")

# Check if there are sufficient data
if steps_per_epoch == 0 or validation_steps == 0:
    raise ValueError("Insufficient data for training or validation. Please provide more images.")

# Fit the model with reduced epochs
model.fit(train_dataset,
          steps_per_epoch=steps_per_epoch,
          epochs=3,  # Reduced number of epochs to prevent running out of data
          validation_data=validation_dataset,
          validation_steps=validation_steps)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(validation_dataset, steps=validation_steps)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(validation_dataset, steps=validation_steps)
print("Predictions shape:", predictions.shape)

# Save the model
model.save('logo_classification_model.keras')