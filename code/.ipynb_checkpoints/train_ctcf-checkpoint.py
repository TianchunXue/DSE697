# train_ctcf.py

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from model_ctcf import BertMultiTaskForCTCF
from dataset_ctcf import CTCFDatasetMultiTask
from metrics import evaluate_binary

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
MAX_EPOCHS = 5
EVAL_FREQ = 100
MAX_LENGTH = 128
LAMBDA_WEIGHT = 0.15
EARLY_STOP = 2
MODEL_NAME = "zhihan1996/DNA_bert_3"

TRAIN_PATH = "../data/train_tmp.jsonl"
VAL_PATH = "../data/eval_tmp.jsonl"
MODEL_SAVE_PATH = "../checkpoints/best_model.pt" 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Initialize model and dataset ---
model = BertMultiTaskForCTCF.from_pretrained(MODEL_NAME, lambda_weight=LAMBDA_WEIGHT).to(device)
train_set = CTCFDatasetMultiTask(TRAIN_PATH, model, max_length=MAX_LENGTH)
val_set = CTCFDatasetMultiTask(VAL_PATH, model, max_length=MAX_LENGTH)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * MAX_EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps)

# --- Training loop ---
best_aupr_token = 0
early_stop_counter = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for step, batch in enumerate(pbar):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        sequence_labels = batch["sequence_labels"].to(device)
        class_type = batch["class_type"]

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            sequence_labels=sequence_labels,
            class_type=class_type
        )

        loss = output["loss"]
        loss.backward()
        optimizer.step()
        scheduler.step()

        pbar.set_postfix(loss=loss.item())

        if (step + 1) % EVAL_FREQ == 0:
            model.eval()
            all_seq_preds, all_seq_true, all_seq_logits = [], [], []
            all_tok_preds, all_tok_true, all_tok_logits = [], [], []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    sequence_labels = batch["sequence_labels"].to(device)
                    class_type = batch["class_type"]

                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=None,
                        sequence_labels=None,
                        class_type=None
                    )

                    # Sequence-level predictions
                    logits_seq = output["logits_sequence"].cpu().numpy()
                    pred_seq = (logits_seq > 0.5).astype(int)
                    true_seq = sequence_labels.cpu().numpy()

                    all_seq_preds.extend(pred_seq)
                    all_seq_true.extend(true_seq)
                    all_seq_logits.extend(logits_seq)

                    # Token-level predictions
                    logits_tok = output["logits_token"].cpu().numpy()
                    pred_tok = (logits_tok > 0.5).astype(int)
                    true_tok = labels.cpu().numpy()

                    # Mask ignored positions (-100)
                    for pred, true, logit in zip(pred_tok, true_tok, logits_tok):
                        mask = true != -100
                        all_tok_preds.extend(pred[mask])
                        all_tok_true.extend(true[mask])
                        all_tok_logits.extend(logit[mask])

            acc_s, pre_s, rec_s, f1_s, auc_s, aupr_s = evaluate_binary(
                np.array(all_seq_preds), np.array(all_seq_true), np.array(all_seq_logits)
            )
            acc_t, pre_t, rec_t, f1_t, auc_t, aupr_t = evaluate_binary(
                np.array(all_tok_preds), np.array(all_tok_true), np.array(all_tok_logits)
            )

            print(f"[Eval] Sequence AUPR={aupr_s:.3f}, Token AUPR={aupr_t:.3f}")

            # Save the best model based on token-level AUPR
            if aupr_t > best_aupr_token:
                print("Saving best model...")
                best_aupr_token = aupr_t
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= EARLY_STOP:
                    print("Early stopping triggered.")
                    break
    if early_stop_counter >= EARLY_STOP:
        break

print("Training complete.")