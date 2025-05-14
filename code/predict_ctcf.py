import torch
from torch.utils.data import DataLoader
from model_ctcf_v3 import BertMultiTaskForCTCF
from dataset_ctcf import CTCFDatasetMultiTask
from metrics import evaluate_binary
import numpy as np
import json

MODEL_PATH = "../checkpoints/best_model_v3.pt"
TEST_PATH = "../data/eval_tmp.jsonl"
MAX_LENGTH = 128
BATCH_SIZE = 32
MODEL_NAME = "zhihan1996/DNA_bert_3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load model
model = BertMultiTaskForCTCF.from_pretrained(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval()

# Load test data
test_dataset = CTCFDatasetMultiTask(TEST_PATH, model, max_length=MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

all_seq_preds, all_seq_true, all_seq_logits = [], [], []
all_tok_preds, all_tok_true, all_tok_logits = [], [], []

with torch.no_grad():
    for batch in test_loader:
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

# Save predictions
with open("../ctcf_test_preds.json", "w") as f:
    for i, (seq, toks) in enumerate(zip(all_seq_preds, all_tok_preds)):
        json.dump({"id": i, "sequence_pred": seq, "token_preds": toks}, f)
        f.write("\n")

# Evaluate performance
acc_s, pre_s, rec_s, f1_s, auc_s, aupr_s = evaluate_binary(
    np.array(all_seq_preds), np.array(all_seq_true), np.array(all_seq_logits))
acc_t, pre_t, rec_t, f1_t, auc_t, aupr_t = evaluate_binary(
    np.array(all_tok_preds), np.array(all_tok_true), np.array(all_tok_logits))

print("[Test Evaluation]")
print(f"Sequence Level: AUPR={aupr_s:.3f}, F1={f1_s:.3f}")
print(f"Token Level:    AUPR={aupr_t:.3f}, F1={f1_t:.3f}")
