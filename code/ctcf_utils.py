# ctcf_utils.py

def kmerize(sequence, k):
    return " ".join([sequence[i:i + k] for i in range(len(sequence) - k + 1)])

def base_to_token_labels(base_labels, k):
    return [
        1 if all(base_labels[i:i + k]) else 0
        for i in range(len(base_labels) - k + 1)
    ]