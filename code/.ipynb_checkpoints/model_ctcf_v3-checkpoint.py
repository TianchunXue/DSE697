import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, BertPreTrainedModel
from ctcf_utils import kmerize

class BertMultiTaskForCTCF(BertPreTrainedModel):
    def __init__(self, config, lambda_weight=0.15, model_name="zhihan1996/DNA_bert_3"):
        super().__init__(config)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(0.2)

        self.token_classifier = nn.Linear(config.hidden_size, 1)
        self.sequence_classifier = nn.Linear(config.hidden_size, 1)

        self.lambda_weight = lambda_weight

        # Use pos_weight for class imbalance
        self.loss_fn_token = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0), reduction='none')
        self.loss_fn_sequence = nn.BCEWithLogitsLoss(reduction='none')

        self.k = int(model_name.split("_")[-1])

    def preprocess(self, sequences, max_length=100):
        kmers = [kmerize(seq, self.k) for seq in sequences]
        return self.tokenizer(
            kmers,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        sequence_output = outputs.last_hidden_state[:, 1:-1, :]
        cls_output = outputs.pooler_output

        token_logits = self.token_classifier(self.dropout(sequence_output)).squeeze(-1)
        seq_logits = self.sequence_classifier(self.dropout(cls_output)).squeeze(-1)

        output_dict = {
            "logits_token": token_logits,
            "logits_sequence": seq_logits
        }

        if labels is not None:
            # Derive sequence-level label from token-level labels (based on strict all-1 match per k-mer)
            sequence_labels = (labels.sum(dim=1) > 0).float()

            token_loss = self.loss_fn_token(token_logits, labels)
            sequence_loss = self.loss_fn_sequence(seq_logits, sequence_labels)

            loss = (1 - self.lambda_weight) * token_loss.mean() + self.lambda_weight * sequence_loss.mean()
            output_dict["loss"] = loss

        return output_dict
