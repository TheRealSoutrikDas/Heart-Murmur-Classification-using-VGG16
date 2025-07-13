# Heart-Murmur-Classification-using-VGG16
A deep learning pipeline that detects **heart murmurs** from **phonocardiogram (PCG)** recordings using **mel-spectrogram images** and a fine-tuned **VGG16** model. The system achieves high performance via robust preprocessing, class balancing, and stratified cross-validation.

---

##  Table of Contents

- 📁 Project Structure
- 🧠 Model Overview
- 🔬 Dataset
- ⚙️ Preprocessing Pipeline
- 🚀 Training & Evaluation
- 🧪 Inference
- 📊 Results Summary
- 📦 Installation
- 📂 Requirements
- 🤝 Contributing
- 📄 License

---

## 📁 Project Structure

```

heart-murmur-classification/
│
├── weights/                    # Saved VGG16 models (per fold)
├── spectrograms_png/          # Preprocessed spectrogram images
├── test_spectrograms/         # PNGs for inference
│
├── run_vgg.py                     # 10-run × 5-fold cross-validation script
├── train_vgg.py               # Single-run 5-fold cross-validation
├── inference.py               # Predict label for a given spectrogram
│
├── spectrogram_metadata.csv   # Labels and paths for all training images
├── vgg16_fold_level_metrics.csv   
├── vgg16_run_level_metrics.csv   # these metric csv are creates after we execute run.py
├── requirements.txt
└── README.md

````

---

## 🧠 Model Overview

| Component     | Details                          |
|---------------|----------------------------------|
| Architecture  | VGG16 (pretrained on ImageNet)   |
| Input         | 224×224 mel-spectrogram PNGs     |
| Output        | Binary: `normal` or `murmur`     |
| Strategy      | Freeze first 5 conv layers       |
| Optimizer     | Adam, LR = 5e-5                  |
| Early Stopping| Patience = 3 (val loss-based)    |
### Model Architecture:
- Freeze first 5 layers
- Replace the last layer by `nn.Linear(4096, 2)`
<img width="1280" height="708" alt="Image" src="https://github.com/user-attachments/assets/86a435c6-3bd1-4896-a919-fe8287156341" />
---

## 🔬 Dataset

- **Source**: [Kaggle - Heart Murmur Classification](https://www.kaggle.com/code/zzettrkalpakbal/heart-murmur-classification/input)
- **Format**: `.wav` files + metadata CSV
- **Classes Used**: `normal`, `murmur`
- **Excluded**: `artifact`, `extrastole`, `unknown`

---

## ⚙️ Preprocessing Pipeline

Each `.wav` file is converted to a 3-second **mel-spectrogram** via:

1. **Resampling** to 4000 Hz
2. **Trimming/Chunking** based on label: normal labeled audio trimmed to 3 seconds while audio with murmur label is split up into 3 second chunks.
3. **Noise Reduction** (via `noisereduce`)
4. **Bandpass Filtering** (25–400 Hz)
5. **Mel-Spectrogram Generation**:
   - 128 mel bands
   - `fmax = 800 Hz`
6. **Image Saving**:
   - `.png`: for VGG16 input
   - `.npy`: (optional) for pipeline speed

---

## 🚀 Training & Evaluation

### 🔁 Cross-Validation Setup

| Feature            | Value       |
|--------------------|-------------|
| Total Runs         | 10          |
| Folds per Run      | 5 (Stratified) |
| Total Models       | 50          |
| Batch Size         | 16          |
| Class Balancing    | WeightedRandomSampler |
| Metrics Averaged   | Across all folds & runs |
| Optimizer          | Adam        |

### 🏷 Metrics Used

- Accuracy
- F1 Score
- Sensitivity (Recall)
- Specificity
- Precision
- Confusion Matrix

| Run	 | F1 Score	| Accuracy	| Sensitivity | Specificity	|Precision|
|--------|----------|-----------|-------------|-------------|--------|
| Run 1	 | 0.8165	| 0.8096	| 0.8226	  | 0.7961	    | 0.8274 |
| Run 2	 | 0.8291	| 0.8192	| 0.8528	  | 0.7843   	| 0.8142 |
| Run 3	 | 0.8530	| 0.8442	| 0.8792	  | 0.8078	    | 0.8304 |
| Run 4	 | 0.8240	| 0.8173	| 0.8415	  | 0.7922   	| 0.8111 |
| Run 5	 | 0.8266	| 0.8212	| 0.8340	  | 0.8078   	| 0.8225 | 
| Run 6	 | 0.8215	| 0.8154	| 0.8340	  | 0.7961   	| 0.8161 |
| Run 7	 | 0.8348	| 0.8288	| 0.8340	  | 0.8235   	| 0.8445 |
| Run 8	 | 0.8289	| 0.8288	| 0.8151	  | 0.8431   	| 0.8506 |
| Run 9	 | 0.8228	| 0.8327	| 0.7660	  | 0.9020  	| 0.8969 |
| Run 10 | 0.8221	| 0.8115	| 0.8377	  | 0.7843  	| 0.8161 |
| Mean	 | 0.8279	| 0.8229	| 0.8317	  | 0.8137  	| 0.8330 |
| Std	 | 0.0102	| 0.0107	| 0.0289	  | 0.0360  	| 0.0260 |
---

## 🧪 Inference

Make predictions using a trained model and a PNG spectrogram.

### Example

```bash
python inference.py
````

In `inference.py`, modify:

```python
IMAGE_PATH = "test_spectrograms/sample.png"
MODEL_PATH = "weights/model_fold_0.pt"
```

Outputs:

```bash
Prediction: murmur (Confidence: 0.9482)
```

---

## 📊 Results Summary

| Metric      | Mean   | Std. Dev. |
| ----------- | ------ | --------- |
| F1 Score    | 0.8279 | ±0.0102   |
| Accuracy    | 0.8229 | ±0.0107   |
| Sensitivity | 0.8317 | ±0.0289   |
| Specificity | 0.8137 | ±0.0360   |
| Precision   | 0.8330 | ±0.0260   |

Saved in:

* `vgg16_fold_level_metrics.csv`
* `vgg16_run_level_metrics.csv`

---

## 📦 Installing Requirements

We recommend using [`uv`](https://github.com/astral-sh/uv) for reproducible and fast Python environments.
### Using uv:
### 🧰 Set up environment

```bash
uv venv
.venv/Script/activate
```

### 📥 Install dependencies

```bash
uv pip install -r requirements.txt
```
### Using pip:
### 🧰 Set up environment

```bash
python -m venv .venv
.venv/Script/activate
```

### 📥 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📂 Requirements

Minimal required packages (CUDA 12.8 compatible):

```txt
audioread==3.0.1
certifi==2025.6.15
cffi==1.17.1
charset-normalizer==3.4.2
colorama==0.4.6
contourpy==1.3.2
cycler==0.12.1
decorator==5.2.1
filelock==3.13.1
fonttools==4.58.5
fsspec==2024.6.1
idna==3.10
jinja2==3.1.4
joblib==1.5.1
kiwisolver==1.4.8
lazy-loader==0.4
librosa==0.11.0
llvmlite==0.44.0
markupsafe==2.1.5
matplotlib==3.10.3
mpmath==1.3.0
msgpack==1.1.1
networkx==3.3
noisereduce==3.0.3
numba==0.61.2
numpy==2.2.6
opencv-python==4.12.0.88
packaging==25.0
pandas==2.3.1
pillow==11.0.0
platformdirs==4.3.8
pooch==1.8.2
pycparser==2.22
pyparsing==3.2.3
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.4
scikit-learn==1.7.0
scipy==1.16.0
seaborn==0.13.2
setuptools==70.2.0
six==1.17.0
soundfile==0.13.1
soxr==0.5.0.post1
sympy==1.13.3
threadpoolctl==3.6.0
torch==2.7.1+cu128
torchaudio==2.7.1+cu128
torchvision==0.22.1+cu128
tqdm==4.67.1
typing-extensions==4.14.1
tzdata==2025.2
urllib3==2.5.0
```
**If Error Installing torch:** remove torch from the requirements.txt file and install it using the pytorch api.
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
---


## 📄 License

This project is licensed under the **MIT License**.

---

## ❤️ Acknowledgements

* Kaggle dataset by Izzet Turkalp Akbasli
* Librosa, PyTorch, VGG16
