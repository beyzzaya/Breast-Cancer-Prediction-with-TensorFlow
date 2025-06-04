# Breast-Cancer-Prediction-with-TensorFlow
This repo contains **` Breast-Cancer-Prediction.ipynb`**, a clear, step-by-step Jupyter notebook that
predicts whether a tumour is malignant or benign using the **Breast Cancer
Wisconsin (Diagnostic)** dataset from `sklearn.datasets`.

### Highlights
- **Data loading:** `load_breast_cancer()` → 30 numeric features.  
- **Pre-processing:** train/validation 80 / 20 split + **StandardScaler**.  
- **Input pipeline:** converts NumPy arrays to a shuffled **tf.data** dataset.  
- **Model:** simple fully connected network  
  `Input → Dense(64, relu) → Dense(1, sigmoid)`.  
- **Training:** SGD optimiser, early stopping on *val_auc* (patience = 25).  
- **Result:** reaches **≈ 0.97 validation AUC** and **≈ 0.95 accuracy** (may vary).
