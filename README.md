# 🧠 Brain Tumor Classification and Growth Prediction using Deep Learning

This repository contains code and documentation for a deep learning-based system designed to classify multiple types of brain tumors and predict their growth using MRI images. The project leverages **transfer learning** models — EfficientNet V2 B0 and ViT-B16 — along with **ensemble methods** for robust classification, and an **LSTM network** to model tumor progression over time.

---

## 📌 Project Objectives

- Classify four types of brain tumors (glioma, meningioma, pituitary tumor, and normal) from MRI images.
- Use transfer learning via EfficientNet V2 B0 and ViT-B16 models.
- Improve classification accuracy with ensemble techniques (simple and weighted average).
- Predict tumor growth patterns using sequential MRI scans and LSTM.

---

## 🧾 Dataset

- **Source**: Brain Tumor MRI Dataset (Kaggle, Masoud Nickparvar 2022)
- **Classes**: Glioma, Meningioma, Pituitary Tumor, Normal
- **Size**: 7,023 contrast-enhanced T1-weighted MRI images
- **Growth Prediction**: Longitudinal scan data from 278 patients with 3+ time points

---

## 🛠️ Technologies Used

- Python 3.9
- PyTorch 1.12
- EfficientNet V2 B0 (via `timm`)
- ViT-B16 (Vision Transformer)
- LSTM (Long Short-Term Memory)
- OpenCV, Pillow, SimpleITK
- scikit-learn, NumPy, Pandas
- TensorBoard, Matplotlib

---

## ⚙️ Methodology

1. **Preprocessing**:
   - Image resizing (224×224), normalization, and augmentation
   - Brain extraction using U-Net
2. **Classification**:
   - Fine-tune EfficientNet V2 B0 and ViT-B16
   - Combine predictions using ensemble methods
3. **Growth Prediction**:
   - Extract deep features + radiomic data from longitudinal scans
   - Input into a Bi-LSTM model to predict volume, diameter, and growth direction

---

## 📊 Results

| Model                       | Accuracy | Top-3 Accuracy | Precision | Recall | F1-score |
|----------------------------|----------|----------------|-----------|--------|----------|
| EfficientNet V2 B0         | 94.2%    | 99.9%          | 94.2%     | 94.2%  | 94.1%    |
| ViT-B16                    | 87.2%    | 99.6%          | 87.7%     | 87.2%  | 87.3%    |
| Average Ensemble           | 93.7%    | 99.9%          | 93.8%     | 93.7%  | 93.7%    |
| Weighted Average Ensemble  | **94.7%**| 99.9%          | 94.1%     | 94.7%  | 94.7%    |
| Geometric Mean Ensemble    | 93.6%    | 99.9%          | 93.7%     | 93.6%  | 93.6%    |

---

## 🧪 Evaluation Metrics

- Classification: Accuracy, Precision, Recall, F1-Score, Top-K Accuracy, Confusion Matrix
- Growth Prediction: MAE, MSE, Cosine Similarity, Pearson Correlation

---

## 📁 Project Structure

📂 brain-tumor-analysis
├── data/
├── models/
│ ├── efficientnet_v2/
│ ├── vit_b16/
│ └── lstm/
├── utils/
├── notebooks/
├── results/
├── train_classification.py
├── train_lstm.py
├── ensemble.py
├── requirements.txt
└── README.md

---

## 🧠 Key Insights

- Combining CNNs (EfficientNet) and Transformers (ViT) enhances model performance.
- Adaptive ensemble weighting improves robustness in heterogeneous MRI data.
- LSTM successfully models temporal tumor progression from MRI sequences.

---

## 🔮 Future Work

- Extend to multi-modal imaging (e.g., PET, CT)
- Integrate patient metadata (age, symptoms, genomics)
- Deploy as a clinical decision support system with UI

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Dataset by Masoud Nickparvar (Kaggle)
- PyTorch, TIMM, HuggingFace libraries
- Inspiration from recent research in neuro-oncology and medical imaging

---

## 📬 Contact

For questions, collaboration, or feedback, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/tejas-singh-10j03/) or open an issue on this repository.

