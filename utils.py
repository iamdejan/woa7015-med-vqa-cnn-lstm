import re
import ssl

import evaluate
import nltk
import torch
from torchvision import transforms

FEATURE_SIZE, WORD_EMBED = 1024, 300
MAX_QU_LEN, NUM_HIDDEN, HIDDEN_SIZE = 30, 2, 512
REGEX = re.compile(r"(\W+)")


train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        # 1. Random Rotation: Rotates image +/- 15 degrees
        transforms.RandomRotation(15),
        # 2. Random Zoom: Zooms in slightly (80% to 100% of image)
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        # 3. Random Lighting: Adjust brightness/contrast (X-rays vary in exposure)
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # NOTE: Do NOT use RandomHorizontalFlip.
        # In medicine, "Left" vs "Right" is crucial. Flipping the image invalidates the label.
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # not every image is in 224x224 size
        transforms.ToTensor(),  # convert to (C,H,W) and [0,1]
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),  # mean=0; std=1
    ]
)


def convert_text_to_token_tensor(text, vocab, max_len=30):
    """Tokenizes and converts question string to indices tensor."""
    # 1. Tokenize (Simple regex split as used in typical VQA)
    # Note: Ensure this matches exactly how you trained the Vocab!
    tokens = [t.strip() for t in REGEX.split(text.lower()) if t.strip()]

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


def image_to_tensor(item, device):
    image = item["image"].convert("RGB")
    image_tensor = train_transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)
    return image_tensor


def question_string_to_tensor(item, question_vocab, device):
    question_str = item["question"]
    question_tensor = convert_text_to_token_tensor(question_str, question_vocab, MAX_QU_LEN)
    question_tensor = question_tensor.unsqueeze(0).to(device)
    return question_tensor


def answer_string_to_tensor(item, answer_vocab, device):
    answer_str = item["answer"].lower().strip()

    # Look up the ID for the WHOLE phrase
    if answer_str in answer_vocab.vocab2idx:
        ans_idx = answer_vocab.word2idx(answer_str)
    else:
        ans_idx = answer_vocab.word2idx("<unk>")

    # Create the target tensor
    # We want a 1D Tensor containing a single class index: [Index]
    answer_tensor = torch.tensor([ans_idx], dtype=torch.long).to(device)
    return answer_tensor


def print_accuracy_report(model, dataset, question_vocab, answer_vocab, is_training: bool, device):
    """
    Evaluates the model and prints a compact one-line report.
    """
    # 1. Setup NLTK (SSL Fix) - Keep this to prevent errors
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    for res in ["wordnet", "punkt", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{res}")
        except LookupError:
            nltk.download(res, quiet=True)

    # 2. Setup Metrics
    model.eval()
    metrics = {"CLOSED": {"correct": 0, "total": 0}, "OPEN": {"correct": 0, "total": 0}, "ALL": {"correct": 0, "total": 0}}
    open_preds_text = []
    open_refs_text = []

    # 3. Evaluation Loop
    for item in dataset:
        with torch.no_grad():
            image = item["image"].convert("RGB")
            image_tensor = val_transform(image).unsqueeze(0).to(device)

            question_str = item["question"]
            question_tensor = convert_text_to_token_tensor(question_str, question_vocab, MAX_QU_LEN)
            question_tensor = question_tensor.unsqueeze(0).to(device)

            logits = model(image_tensor, question_tensor)
            pred_idx = torch.argmax(logits, dim=1).item()
            pred_word = answer_vocab.idx2word(pred_idx)

            ground_truth = item["answer"].lower().strip()

            if ground_truth in ["yes", "no"]:
                q_type = "CLOSED"
            else:
                q_type = "OPEN"
                open_preds_text.append(pred_word)
                open_refs_text.append(ground_truth)

            metrics["ALL"]["total"] += 1
            metrics[q_type]["total"] += 1

            if pred_word == ground_truth:
                metrics["ALL"]["correct"] += 1
                metrics[q_type]["correct"] += 1

    # 4. COMPACT PRINT LOGIC
    stats = []

    # Format Accuracy: "Closed: [percentage]% | Open: [percentage]% | All: [percentage]%"
    for key in ["CLOSED", "OPEN", "ALL"]:
        if metrics[key]["total"] > 0:
            acc = (metrics[key]["correct"] / metrics[key]["total"]) * 100
            stats.append(f"{key}: {acc:.2f}%")
        else:
            stats.append(f"{key}: N/A")

    # Format Semantic Metrics: "Meteor: [percentage]% | Rouge: [percentage]%"
    if len(open_preds_text) > 0:
        meteor = evaluate.load("meteor")
        meteor_score = meteor.compute(predictions=open_preds_text, references=open_refs_text)
        stats.append(f"METEOR: {(meteor_score['meteor'] * 100):.2f}%")

        rouge = evaluate.load("rouge")
        rouge_score = rouge.compute(predictions=open_preds_text, references=open_refs_text)
        stats.append(f"ROUGE-L: {(rouge_score['rougeL'] * 100):.2f}%")

    # Print One-Liner
    print(f">> [Eval] {' | '.join(stats)}")

    if is_training:
        model.train()  # Switch back to train mode for next epoch
