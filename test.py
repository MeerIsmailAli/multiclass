from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('logo_classification_model.keras')

# Assuming class1 and class2 are the class names (update with actual class names)
class_indices = {'class1': 0, 'class2': 1, 'class3': 2, 'class4': 3, 'class5': 4}  # Example class indices

# Define image size (replace with the values used during training)
img_height = 50
img_width = 50

# Load and preprocess the image
image_path = '/home/meer/Desktop/multiclass/multiclass/new_data/1200px-Mercedes_Benz_Logo_11.jpg'  # Replace with your image path
img = load_img(image_path, target_size=(img_height, img_width))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)  # Add batch dimension
x = x / 255.0  # Normalize pixel values

# Make predictions
predictions = model.predict(x)

# Get the predicted class probabilities
predicted_probs = predictions[0]

# Get the predicted class index
predicted_class_idx = np.argmax(predicted_probs)

# Map the predicted class index to the class name
class_names = list(class_indices.keys())
predicted_class = class_names[predicted_class_idx]

print(f'Predicted class: {predicted_class}')
print(f'Predicted probabilities: {predicted_probs}')
