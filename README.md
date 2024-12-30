# **Tuberculosis Detection Using Deep Learning**

This repository contains the implementation of a deep learning pipeline for detecting tuberculosis (TB) from chest X-ray images. It uses a pre-trained ResNet50 model fine-tuned for binary classification and includes various techniques for handling class imbalance and optimizing performance.

---

## **Project Overview**

Tuberculosis detection from chest X-rays is a critical task in healthcare. This project provides a complete pipeline for training, evaluating, and deploying a machine learning model to classify chest X-ray images into:
1. **Normal (Class 0)**
2. **TB Positive (Class 1)**

The project structure supports:
- **Data Preprocessing**: Handles X-ray images, lung segmentation masks, and metadata.
- **Model Training and Saving**: Includes weighted loss and threshold tuning for performance optimization.
- **Evaluation**: Metrics like precision, recall, F1-score, and confusion matrix for assessing model performance.

---

## **Project Structure**

```
TB-DETECTION/
│
├── datasets/
│   ├── image/               # X-ray images
│   ├── mask/                # Lung segmentation masks
│   └── MetaData.csv         # Clinical metadata
│
├── .gitignore               # Ignored files for Git
├── best_model.pth           # Best-performing model weights
├── main.ipynb               # Jupyter notebook containing the full pipeline
├── README.md                # Project documentation
├── tb_detection_model.pth   # Final trained model weights
```

---

## **Getting Started**

### **Prerequisites**
- Python 3.7+
- PyTorch
- torchvision
- matplotlib
- pandas
- scikit-learn

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/tb-detection.git
   cd tb-detection
   ```
2. Install the required dependencies:
   ```bash
   pip install torch torchvision matplotlib pandas scikit-learn
   ```

---

## **Dataset**

The dataset consists of 704 chest X-ray images from:
- **Montgomery County Chest X-ray Database (USA)**
- **Shenzhen Chest X-ray Database (China)**

Each X-ray is labeled as **Normal** (Class 0) or **TB Positive** (Class 1). Additionally, lung segmentation masks and clinical metadata (e.g., age, gender) are provided.

---

## **How to Run**

### **1. Train the Model**
1. Place the dataset in the `datasets/` directory as shown above.
2. Open the `main.ipynb` notebook and execute the cells in sequence to:
   - Preprocess data.
   - Train the model.
   - Save the best model as `best_model.pth`.

### **2. Evaluate the Model**
- Use the provided evaluation function in `main.ipynb` to assess the model's performance on the test set.

### **3. Threshold Tuning**
- Experiment with different thresholds (e.g., `0.3` to `0.7`) to optimize the precision-recall trade-off.

---

## **Results**

- **Best Accuracy**: **80%** with class weights `[1.0, 3.0]` and threshold `0.5`.
- **Evaluation Metrics**:
  - **Normal (Class 0)**: Precision = 0.75, Recall = 0.92, F1-Score = 0.82
  - **TB Positive (Class 1)**: Precision = 0.89, Recall = 0.69, F1-Score = 0.77

---

## **Customization**

### **Change Dataset Path**
Update the dataset paths in `main.ipynb`:
```python
image_dir = '/datasets/image/'
mask_dir = '/datasets/mask/'
metadata_path = '/datasets/MetaData.csv'
```

### **Threshold Tuning**
Adjust thresholds in the evaluation step:
```python
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
for t in thresholds:
    evaluate_with_threshold(model, test_loader, threshold=t)
```

---

## **Future Improvements**
1. **Incorporate Metadata**: Use demographic features (age, gender) as additional input to the model.
2. **Visualization**: Add Grad-CAM visualizations for model explainability.
3. **Enhanced Loss Function**: Experiment with focal loss for better handling of class imbalance.

