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

For this project, we used the [Dataset Name], which is commonly used for [task type, e.g., image classification, text classification, etc.]. The dataset contains [number of instances] of data, with each instance consisting of [describe data features, such as text, images, or numerical data]. The dataset is publicly available at [provide dataset link, if applicable].

### **Dataset Details:**
- **Number of Samples:** [Total number of samples]
- **Features:** [Describe key features of the dataset]
- **Target Variable:** [Describe the target variable — what you're trying to predict/classify]
- **Classes (if applicable):** [List the classes or categories, if classification task]
  
**Data Source:** [Dataset URL or citation]

---

## **Data Preparation**

### **Data Preprocessing**

To prepare the data for model training, we performed the following preprocessing steps:

1. **Cleaning:** [Explain any data cleaning steps, e.g., handling missing values, removing duplicates]
2. **Normalization/Standardization:** [If applicable, explain any normalization or standardization methods used to scale data]
3. **Splitting the Dataset:** The dataset was split into training and testing sets with a ratio of [80%/20% or other split ratio].
4. **Encoding (if necessary):** [Describe any encoding steps, such as one-hot encoding for categorical data]

---

## **Data Augmentation**

To improve the generalization of our model, we applied data augmentation techniques:

1. **[Type of Augmentation 1]**: [Description of augmentation, e.g., random rotation, flipping for image data, etc.]
2. **[Type of Augmentation 2]**: [Further augmentations applied, such as noise injection, scaling, etc.]

These techniques helped improve the robustness of our models by increasing the diversity of the training data.

---

## **Model Implementation**

We implemented two different machine learning models to evaluate their performance on the given task.

### **Model 1: [Model Name, e.g., Convolutional Neural Network (CNN)]**

- **Architecture:**  
  The model architecture consists of [briefly describe the layers used, e.g., convolutional layers, fully connected layers, activation functions, etc.].  
  - [Layer 1: Description, e.g., Conv2D, filters=32, kernel size=3x3]
  - [Layer 2: Description, e.g., MaxPooling2D, pool size=2x2]
  - [Layer 3: Description, e.g., Dense, units=128, activation=ReLU]

- **Hyperparameters:**
  - Learning Rate: [value]
  - Batch Size: [value]
  - Epochs: [value]

- **Model Summary:**
  ```python
  model.summary()  # Display the model architecture summary
### **Model 2: [Model Name, e.g., Support Vector Machine (SVM)]**

- **Architecture:**  
  The model uses a traditional machine learning approach. It consists of [describe the architecture, e.g., hyperparameters used in the SVM model, kernel function, etc.].

- **Hyperparameters:**
  - Kernel: [Linear/Polynomial/RBF]
  - C: [Value of C]
  - Gamma: [Value of Gamma]

- **Model Training:**  
  The model was trained using [optimizer, if applicable], and the classifier output was evaluated using [accuracy, F1-score, etc.].

---

## **Model Evaluation**

We evaluated the performance of both models using the following metrics:

- **Accuracy:** [Value for Model 1], [Value for Model 2]
- **Precision:** [Value for Model 1], [Value for Model 2]
- **Recall:** [Value for Model 1], [Value for Model 2]
- **F1 Score:** [Value for Model 1], [Value for Model 2]

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

1. [Paper Title and Citation]  
2. [Additional References, e.g., datasets, libraries used, etc.]

---

