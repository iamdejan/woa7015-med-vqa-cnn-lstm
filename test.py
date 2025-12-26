import re

import evaluate
import torch
from datasets import load_dataset
from torchvision import transforms

from build_vocab import Vocab
from model import VQAModel

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


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    test_dataset = dataset["test"]

    question_vocab = Vocab("./data/test/q_vocab.json")
    answer_vocab = Vocab("./data/test/ans_vocab.json")

    model = VQAModel(FEATURE_SIZE, question_vocab.vocab_size, answer_vocab.vocab_size, WORD_EMBED, HIDDEN_SIZE, NUM_HIDDEN)
    model.eval()

    all_preds = []
    all_refs = []

    for item in test_dataset:
        with torch.no_grad():
            # --- PREPROCESS IMAGE ---
            image = item["image"].convert("RGB")  # Ensure 3 channels
            # transform returns (3, 224, 224)
            transformed_image = transform(image)
            # Add Batch Dimension: (1, 3, 224, 224) because VGG only accepts image with batch dimension
            transformed_image = transformed_image.unsqueeze(0)

            # --- PREPROCESS QUESTION ---
            question_str = item["question"]
            # Convert string -> Tensor of indices
            question_tensor = process_question(question_str, question_vocab, MAX_QU_LEN)
            # Add Batch Dimension: (1, 30)
            question_tensor = question_tensor.unsqueeze(0)

            # --- FORWARD PASS ---
            logits = model(transformed_image, question_tensor)

            # --- PREDICTION ---
            # LogSoftmax + Argmax
            log_probs = torch.log_softmax(logits, dim=1)
            pred_idx = torch.argmax(log_probs, dim=1).item()  # .item() to get int

            # Convert Index back to Word
            pred_word = answer_vocab.idx2word(pred_idx)
            ground_truth = str(item["answer"]).lower()

            print(f"Q: {question_str}")
            print(f"A (Predicted): {pred_word}")
            print(f"A (Ground Truth): {ground_truth}")
            print("-" * 20)

            # Store for metrics
            all_preds.append(pred_word)
            all_refs.append(ground_truth)

    # 3. Initialize Metrics
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")

    # BLEU
    # BLEU requires references to be a list of lists: [['yes'], ['no'], ['abcdef']]
    bleu_refs = [[r] for r in all_refs]
    results_bleu = bleu_metric.compute(predictions=all_preds, references=bleu_refs)
    print(f"BLEU: {results_bleu['bleu']:.4f}")

    # ROUGE-L
    # ROUGE expects list of strings: ['yes', 'no', 'abcdef']
    results_rouge = rouge_metric.compute(predictions=all_preds, references=all_refs)
    print(f"ROUGE-L: {results_rouge['rougeL']:.4f}")

    # METEOR
    results_meteor = meteor_metric.compute(predictions=all_preds, references=all_refs)
    print(f"METEOR: {results_meteor['meteor']:.4f}")


if __name__ == "__main__":
    main()
