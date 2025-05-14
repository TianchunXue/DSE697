import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import numpy as np
import json
from model_ctcf_v3 import BertMultiTaskForCTCF
from dataset_ctcf import CTCFDatasetMultiTask
from metrics import evaluate_binary

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm

MODEL_NAME = "zhihan1996/DNA_bert_3"
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5
K = 5
LAMBDA_WEIGHT = 0.15

DATA_PATH = "../data/merged_ctcf_dataset.jsonl"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]

def run_fold(train_lines, val_lines, fold_idx):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertMultiTaskForCTCF.from_pretrained(MODEL_NAME, lambda_weight=LAMBDA_WEIGHT).to(device)

    with open("train_fold.jsonl", "w") as f:
        for line in train_lines:
            f.write(json.dumps(line) + "\n")
    with open("val_fold.jsonl", "w") as f:
        for line in val_lines:
            f.write(json.dumps(line) + "\n")

    train_set = CTCFDatasetMultiTask("train_fold.jsonl", model, max_length=MAX_LENGTH)
    val_set = CTCFDatasetMultiTask("val_fold.jsonl", model, max_length=MAX_LENGTH)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * len(train_loader) * EPOCHS,
        num_training_steps=len(train_loader) * EPOCHS)

    best_aupr = 0
    model.train()
    for epoch in range(EPOCHS):
        pbar = tqdm(train_loader, desc=f"Fold {fold_idx+1} Epoch {epoch+1}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"][:, 1:-1].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())

    # Evaluation
    model.eval()
    all_seq_preds, all_seq_true, all_seq_logits = [], [], []
    all_tok_preds, all_tok_true, all_tok_logits = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"][:, 1:-1].to(device)

            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            seq_logits = output["logits_sequence"]
            tok_logits = output["logits_token"]

            seq_pred = (seq_logits > 0.5).int().tolist()
            tok_pred = (tok_logits > 0.5).int().cpu().tolist()

            all_seq_preds.extend(seq_pred)
            all_seq_true.extend((labels.sum(dim=1) > 0).int().cpu().tolist())
            all_seq_logits.extend(seq_logits.cpu().tolist())

            for pred, true, logit in zip(tok_pred, labels.cpu().tolist(), tok_logits.cpu().tolist()):
                all_tok_preds.extend(pred)
                all_tok_true.extend(true)
                all_tok_logits.extend(logit)

    acc_s, pre_s, rec_s, f1_s, auc_s, aupr_s = evaluate_binary(
        np.array(all_seq_preds), np.array(all_seq_true), np.array(all_seq_logits))
    acc_t, pre_t, rec_t, f1_t, auc_t, aupr_t = evaluate_binary(
        np.array(all_tok_preds), np.array(all_tok_true), np.array(all_tok_logits))

    print(f"[Fold {fold_idx+1}] Seq AUPR={aupr_s:.3f}, Token AUPR={aupr_t:.3f}")
    return aupr_s, f1_s, aupr_t, f1_t

if __name__ == "__main__":
    all_data = load_jsonl(DATA_PATH)
    kf = KFold(n_splits=K, shuffle=True, random_state=42)
    
    seq_auprs, seq_f1s, tok_auprs, tok_f1s = [], [], [], []

    for i, (train_idx, val_idx) in enumerate(kf.split(all_data)):
        train_lines = [all_data[j] for j in train_idx]
        val_lines = [all_data[j] for j in val_idx]

        aupr_s, f1_s, aupr_t, f1_t = run_fold(train_lines, val_lines, i)
        seq_auprs.append(aupr_s)
        seq_f1s.append(f1_s)
        tok_auprs.append(aupr_t)
        tok_f1s.append(f1_t)

    print("\n===== Cross-Validation Summary =====")
    print(f"Sequence Level AUPR: {np.mean(seq_auprs):.3f} ± {np.std(seq_auprs):.3f}")
    print(f"Sequence Level F1:   {np.mean(seq_f1s):.3f} ± {np.std(seq_f1s):.3f}")
    print(f"Token Level AUPR:    {np.mean(tok_auprs):.3f} ± {np.std(tok_auprs):.3f}")
    print(f"Token Level F1:      {np.mean(tok_f1s):.3f} ± {np.std(tok_f1s):.3f}")
