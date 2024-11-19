# DSL 501: Machine Learning Course Project

**Project Title:** Implementation of 'DFN-PSAN: Multi-level deep information feature fusion extraction network for interpretable plant disease classification'

**Course:** DSL 501 - Machine Learning  
 
**Team Member:**  
- Niladri Biswas, Roll No: 12310870

---

## **Project Summary**

In this project, we implemented the machine learning model described in the paper titled "DFN-PSAN: Multi-level deep information feature fusion extraction network for interpretable plant disease classification". The goal of this project is to [describe the primary objective of the paper — for example, predict a certain phenomenon, classify objects, etc.]. The paper proposes a [type of model used in the paper], and in this implementation, we aim to replicate the proposed model and evaluate its performance using a specific dataset.

### **Paper Summary**

The paper discusses the use of a multi-level deep information feature fusion extraction network for interpretable plant disease classification. It highlights the importance of combining deep feature extraction with semantic attention mechanisms to improve both classification accuracy and model interpretability. The authors employed a DFN-PSAN (Deep Feature Network with Plant Disease Semantic Attention Network) to achieve high accuracy in identifying plant diseases from images. The model's primary contributions include the integration of multi-level feature fusion and semantic attention mechanisms, allowing the network to focus on relevant regions in images and providing more interpretable results in plant disease classification.

---

## **Dataset**

### **Dataset Overview**

For this project, we used the Plant Diseases Detection Datasets. The dataset contains 87K rgb images of healthy and diseased crop leaves of data  which is categorized into 38 different classes. The total dataset is divided into 80/20 ratio of training and validation. A directory contains 33 test images for prediction purpose.

### **Dataset Details:**
- **Number of Samples:** 87K rgb images
- **Features:** 14 unique plant leaves ['Apple', 'Blueberry', 'Cherry_(including_sour)', 'Corn_(maize)', 'Grape', 'Orange', 'Peach', 'Pepper,_bell', 'Potato', 'Raspberry', 'Soybean', 'Squash', 'Strawberry', 'Tomato']
- **Target Variable:** healthy / diseased crop leaves
- **Classes (if applicable):** 38 classes ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___healthy', 'Corn_(maize)___Northern_Leaf_Blight', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___healthy', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___healthy', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_mosaic_virus', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus']
  
**Data Source:** https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data

---

## **Data Preparation**

### **Data Preprocessing**

To prepare the data for model training, we performed the following preprocessing steps:

1. **Cleaning:** The dataset was assumed to be clean, with no missing values or duplicates. Images were verified to ensure they were properly labeled and placed in corresponding class directories.
2. **Normalization/Standardization:** The pixel values of the images were normalized by scaling them to the range [0, 1] using the rescale=1./255 parameter in the ImageDataGenerator. This ensures that all input features have a consistent scale and helps with faster convergence during training.
3. **Splitting the Dataset:** The dataset was split into 80/20 training and validation sets. The "train" directory has 70243 images belonging to 38 classes , while the "valid" directory has 17557 images belonging to 38 classes.
4. **Encoding (if necessary):** Since the model is designed for multi-class classification, the labels were automatically one-hot encoded using class_mode='categorical' in the ImageDataGenerator. This ensures that the output labels for each image are represented as a one-hot vector corresponding to the class of the image.

---

## **Data Augmentation**

To improve the generalization of our model, we applied data augmentation techniques:

1. **[Model 1 Augmentation]**:     rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
2. **[Model 2 Augmentation]**:     rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'

These techniques helped improve the robustness of our models by increasing the diversity of the training data.

---

## **Model Implementation**

We implemented two different machine learning models to evaluate their performance on the given task.

### **Model 1: [DFN-PSAN (Deep Feature Network with Pyramid Spatial Attention Network)]**

- **Architecture:**  
 The model architecture consists of multiple convolutional blocks with different kernel sizes, attention mechanisms, feature fusion, and global average pooling followed by fully connected layers.

  [Layer 1: Conv2D, filters=32, kernel size=3x3, activation=ReLU, padding='same']  
  [Layer 2: MaxPooling2D, pool size=2x2]  
  [Layer 3: Conv2D, filters=64, kernel size=3x3, activation=ReLU, padding='same']  
  [Layer 4: MaxPooling2D, pool size=2x2]  
  [Layer 5: Conv2D, filters=32, kernel size=5x5, activation=ReLU, padding='same']  
  [Layer 6: MaxPooling2D, pool size=2x2]  
  [Layer 7: Conv2D, filters=64, kernel size=5x5, activation=ReLU, padding='same']  
  [Layer 8: MaxPooling2D, pool size=2x2]  
  [Layer 9: Reshape, target shape=(-1, 64)]  
  [Layer 10: Attention, input=[x1_reshaped, x1_reshaped]]  
  [Layer 11: Attention, input=[x2_reshaped, x2_reshaped]]  
  [Layer 12: Reshape, target shape=(x1.shape[1], x1.shape[2], 64)]  
  [Layer 13: Reshape, target shape=(x2.shape[1], x2.shape[2], 64)]  
  [Layer 14: Concatenate, axis=-1]  
  [Layer 15: GlobalAveragePooling2D]  
  [Layer 16: Dense, units=128, activation=ReLU]  
  [Layer 17: Dense, units=64, activation=ReLU]  
  [Layer 18: Dense, units=num_classes, activation=Softmax]  

- **Hyperparameters:**
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 10

- **Model Summary:**
  - Total Layers: 18
  - Trainable Parameters: 100,326 (391.90 KB)
  - Optimizer: Adam
  - Loss: categorical_crossentropy
  - Metrics: accuracy
### **Model 2: [CNN with Squeeze-and-Excitation (SE) Block]**

- **Architecture:**  
  The model architecture consists of convolutional layers with varying kernel sizes, followed by the Squeeze-and-Excitation (SE) block for feature recalibration, global average pooling, and dense layers for classification.

  [Layer 1: Input] Shape: (256, 256, 3)  
  [Layer 2: Conv2D] Filters: 64, Kernel size: 3x3, Activation: ReLU, Padding: 'same'  
  [Layer 3: Conv2D] Filters: 128, Kernel size: 3x3, Activation: ReLU, Padding: 'same'  
  [Layer 4: Conv2D] Filters: 256, Kernel size: 3x3, Activation: ReLU, Padding: 'same'  
  [Layer 5: Conv2D] Filters: 64, Kernel size: 3x3, Activation: ReLU, Padding: 'same'  
  [Layer 6: Conv2D] Filters: 64, Kernel size: 5x5, Activation: ReLU, Padding: 'same'  
  [Layer 7: Conv2D] Filters: 64, Kernel size: 7x7, Activation: ReLU, Padding: 'same'  
  [Layer 8: Conv2D] Filters: 64, Kernel size: 9x9, Activation: ReLU, Padding: 'same'  
  [Layer 9: Add] Operation: Sum of the feature maps from layers 6, 7, and 8  
  [Layer 10: GlobalAveragePooling2D] Operation: Global average pooling  
  [Layer 11: Reshape] Target shape: (1, 1, 64)  
  [Layer 12: SEBlock] Operation: Squeeze-and-Excitation block (attention mechanism)  
  [Layer 13: Flatten] Operation: Flatten the output for dense layers  
  [Layer 14: Dense] Units: num_classes (38), Activation: Softmax  

- **Hyperparameters:**
  - Learning Rate: 0.001
  - Batch Size: 32
  - Epochs: 10

- **Model Summary:**  
  - Total Layers: 14
  - Trainable Parameters: 1,156,458 (4.41 MB)
  - Optimizer: Adam
  - Loss: categorical_crossentropy
  - Metrics: accuracy

---

## **Model Evaluation**

We evaluated the performance of both models using the following metrics:

- **Model 1:** 
  - Training Accuracy: 0.8062  Training Loss: 0.6183
  - Validation Accuracy: 0.8087 Validation Loss: 0.6047 

- **Model 2:** 
  - Training Accuracy: 0.8269  Training Loss: 0.5730
  - Validation Accuracy: 0.8300 Validation Loss: 0.5595 

### **Training and Validation Graphs**

Below are the graphs showing the model performance during training:

1. **Model 1 (e.g., CNN) - Loss and Accuracy Curves**  
   ![Model 1 - Loss vs Accuracy](./images/model1_loss_accuracy.png)

2. **Model 2 (e.g., SVM) - Confusion Matrix and Performance**  
   ![Model 2 - Confusion Matrix](./images/model2_confusion_matrix.png)

---

## **Results and Analysis**

Both models were trained and evaluated on the dataset, and we compared their performance based on the evaluation metrics.

- **Model 1 (CNN)** performed with an accuracy of [value]% and showed strong results in [tasks, e.g., image classification, object detection].
- **Model 2 (SVM)** performed with an accuracy of [value]% and was better suited for [reason or task, e.g., smaller datasets, simpler problems].

### **Observations:**
- [Highlight any interesting observations, such as overfitting/underfitting, or any other findings from the training process.]
- [Comparison between models, e.g., why one performed better than the other.]

---

## **Conclusion**

In conclusion, we successfully replicated the model described in the paper "[Paper Title]" and evaluated its performance on the given dataset. After experimenting with two different models, we found that [which model performed better and why]. Future work could explore [areas for improvement, such as hyperparameter tuning, using different augmentation techniques, or trying other models].

---

## **References**

1. Paper : https://www.sciencedirect.com/science/article/pii/S0168169923008694
---

