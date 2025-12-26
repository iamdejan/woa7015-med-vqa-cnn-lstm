import evaluate
import torch
from datasets import load_dataset

import utils
from build_vocab import Vocab
from model import VQAModel


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    test_dataset = dataset["test"]

    question_vocab = Vocab("./data/test/q_vocab.json")
    answer_vocab = Vocab("./data/test/ans_vocab.json")

    model = VQAModel(utils.FEATURE_SIZE, question_vocab.vocab_size, answer_vocab.vocab_size, utils.WORD_EMBED, utils.HIDDEN_SIZE, utils.NUM_HIDDEN)
    model.eval()

    all_preds = []
    all_refs = []

    for item in test_dataset:
        with torch.no_grad():
            # --- PREPROCESS IMAGE ---
            image = item["image"].convert("RGB")
            image_tensor = utils.transform(image).unsqueeze(0)

            # --- PREPROCESS QUESTION ---
            question_str = item["question"]
            # Convert string -> Tensor of indices
            question_tensor = utils.convert_text_to_token_tensor(question_str, question_vocab, utils.MAX_QU_LEN)
            # Add Batch Dimension: (1, 30)
            question_tensor = question_tensor.unsqueeze(0)

            # --- FORWARD PASS ---
            logits = model(image_tensor, question_tensor)

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
