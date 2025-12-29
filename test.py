import ssl

import nltk
import torch
from datasets import load_dataset

import utils
from build_vocab import Vocab
from model import VQAModel

device = torch.device("cpu")


def main():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("omw-1.4")

    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    test_dataset = dataset["test"]

    question_vocab = Vocab("./data/q_vocab.json")
    answer_vocab = Vocab("./data/ans_vocab.json")

    model = VQAModel(
        feature_size=utils.FEATURE_SIZE,
        qu_vocab_size=question_vocab.vocab_size,
        ans_vocab_size=answer_vocab.vocab_size,
        word_embed=utils.WORD_EMBED,
        hidden_size=utils.HIDDEN_SIZE,
        num_hidden=utils.NUM_HIDDEN,
    )
    model.load_state_dict(torch.load("checkpoint/best_model.pt", weights_only=True))
    model.eval()

    # Storage for detailed metrics
    metrics = {"CLOSED": {"correct": 0, "total": 0}, "OPEN": {"correct": 0, "total": 0}, "ALL": {"correct": 0, "total": 0}}

    open_preds_text = []
    open_refs_text = []

    print(f"Start Testing on {len(test_dataset)} samples...")
    for item in test_dataset:
        with torch.no_grad():
            image_tensor = utils.image_to_tensor(item, device)
            question_tensor = utils.question_string_to_tensor(item, question_vocab, device)

            # --- FORWARD PASS ---
            logits = model(image_tensor, question_tensor)

            # --- PREDICTION ---
            # LogSoftmax + Argmax
            log_probs = torch.log_softmax(logits, dim=1)
            pred_idx = torch.argmax(log_probs, dim=1).item()  # .item() to get int

            # Convert Index back to Word
            pred_word = answer_vocab.idx2word(pred_idx)
            ground_truth = str(item["answer"]).lower().strip()
            if ground_truth in ["yes", "no"]:
                q_type = "CLOSED"
            else:
                q_type = "OPEN"
                # Store text for semantic evaluation later
                open_preds_text.append(pred_word)
                open_refs_text.append(ground_truth)

            # --- Update Accuracy Counters ---
            metrics["ALL"]["total"] += 1
            metrics[q_type]["total"] += 1

            if pred_word == ground_truth:
                metrics["ALL"]["correct"] += 1
                metrics[q_type]["correct"] += 1

    utils.print_accuracy_report(model, test_dataset, question_vocab, answer_vocab, False, device)


if __name__ == "__main__":
    main()
