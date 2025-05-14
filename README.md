# CTCF Binding Site Prediction (DNABERT Fine-Tuning)

This repository contains the code for multi-task fine-tuning of [DNABERT](https://github.com/jerryji1993/DNABERT) to predict CTCF binding sites at both the sequence and base resolution.

---

## 📁 Project Structure

```
.
├── train_ctcf_v3.py            # Fine-tuning script for DNABERT model
├── model_ctcf_v3.py            # Multi-task model definition (sequence + token prediction)
├── dataset_ctcf.py             # Dataset class for loading and tokenizing input sequences
├── predict_ctcf.py             # Predict on test set with ground truth
├── predict_ctcf_new_data.py    # Predict on independent dataset
├── cross_valid_ctcf.py         # 5-fold cross-validation pipeline
├── metrics.py                  # AUPR, F1 evaluation metrics
├── ctcf_utils.py               # Helper functions (e.g., k-mer label construction)
```

---

## 🧪 Task Description

We train a DNABERT-based model to perform **multi-task prediction**:

* **Sequence-level**: Whether a 100bp sequence contains a CTCF binding site.
* **Token-level**: Which bases in the sequence correspond to the CTCF motif.

---

## 🧪 Training

```bash
python train_ctcf_v3.py
```

Make sure your input data (`train_tmp.jsonl`, `eval_tmp.jsonl`) follows the structure:

```json
{
  "sequence": "ATCG...",
  "labels": [0, 1, 1, 1, 0, ...]  // base-level binary labels
}
```

---

## 📊 Evaluation

Run predictions and calculate performance:

```bash
python predict_ctcf.py
```

Test on independent data:

```bash
python predict_ctcf_new_data.py
```

Perform 5-fold cross-validation:

```bash
python cross_valid_ctcf.py
```

---

## 📈 Metrics

Evaluation includes:

* **Sequence-level**: AUPR & F1
* **Token-level**: AUPR & F1

Metrics are implemented in `metrics.py`.

---
