# Kaggle Submission Guide

This directory contains the tools and files necessary to prepare your agent for submission to the Kaggle ConnectX competition.

## ⚠️ Important Note

Kaggle ConnectX competition requires a **single python file** submission. Since our agents (especially AlphaZero and Rainbow DQN) rely on trained model weights (`.pth` files), we cannot simply submit the code.

**We must embed the model weights directly into the submission script.**

## 🛠️ Preparation Tool

We provide a script `prepare_submission.py` that handles this process automatically. It performs the following steps:

1.  **Reads** your trained model file (`.pth`).
2.  **Compresses** the model weights using `gzip`.
3.  **Encodes** the compressed data into a Base64 string.
4.  **Embeds** this string into a template script.
5.  **Generates** a final `main.py` that decodes and loads the model at runtime.

## 🚀 How to Create a Submission

### 1. Identify your best model

Locate the `.pth` file you want to submit. For example: `submission/alpha-zero-v1.pth`.

### 2. Run the preparation script

Run the following command from the `submission` directory:

```bash
python prepare_submission.py --model <path_to_your_model> --output main.py
```

**Example:**

```bash
python prepare_submission.py --model alpha-zero-v1.pth --output main.py
```

### 3. Submit to Kaggle

Upload the generated `main.py` file to the Kaggle competition page.

## 📂 File Structure

- `prepare_submission.py`: The script that embeds the model and generates the submission file.
- `main.py`: The generated submission file (do not edit manually).
- `*.pth`: Your trained model checkpoints.

## 🔍 Verification

Before submitting, you can verify the generated `main.py` works locally by running it or importing it in the playground.

```bash
# Test if it runs without error
python main.py
```
