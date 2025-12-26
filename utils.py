import re

import torch
from torchvision import transforms

FEATURE_SIZE, WORD_EMBED = 1024, 300
MAX_QU_LEN, NUM_HIDDEN, HIDDEN_SIZE = 30, 2, 512
REGEX = re.compile(r"(\W+)")


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # not every image is in 224x224 size
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean=0; std=1
    ]
)


def process_question(question_str, vocab, max_len=30):
    """Tokenizes and converts question string to indices tensor."""
    # 1. Tokenize (Simple regex split as used in typical VQA)
    # Note: Ensure this matches exactly how you trained the Vocab!
    tokens = [t.strip() for t in REGEX.split(question_str.lower()) if t.strip()]

    # 2. Convert to Indices
    indices = [vocab.word2idx(token) for token in tokens]

    # 3. Pad or Truncate
    if len(indices) < max_len:
        # Pad with 0 (assuming 0 is <pad> in your vocab)
        indices += [0] * (max_len - len(indices))
    else:
        indices = indices[:max_len]

    # 4. Convert to Tensor
    return torch.tensor(indices, dtype=torch.long)
