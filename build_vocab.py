import json
import os
import re

from datasets import load_dataset

# Create directory if not exists
if not os.path.exists("./data"):
    os.makedirs("./data")


class Vocab:
    def __init__(self, vocab_file):
        self.vocab = self.load_vocab(vocab_file)
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def load_vocab(self, vocab_file):
        with open(vocab_file) as f:
            ls = json.load(f)
        return ls

    def word2idx(self, vocab):
        return self.vocab2idx.get(vocab, self.vocab2idx["<unk>"])

    def idx2word(self, idx):
        return self.vocab[idx]


REGEX = re.compile(r"(\W+)")


def build_vocab(dataset):
    # Collect ALL unique questions and answers from BOTH train and test
    all_questions = []
    all_answers = []

    # Iterate over both splits
    for split in ["train", "test"]:
        for item in dataset[split]:
            all_questions.append(item["question"])
            all_answers.append(item["answer"])

    # --- Build Question Vocab ---
    q_vocab = []
    for q in all_questions:
        split_q = REGEX.split(q.lower())
        q_vocab.extend([w.strip() for w in split_q if len(w.strip()) > 0])

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab.insert(0, "<pad>")
    q_vocab.insert(1, "<unk>")

    with open("./data/q_vocab.json", "w") as f:
        json.dump(q_vocab, f, indent=4)
    print(f"Saved {len(q_vocab)} question tokens.")

    # --- Build Answer Vocab ---
    ans_vocab = []
    for a in all_answers:
        clean_ans = a.lower().strip()
        if len(clean_ans) > 0:
            ans_vocab.append(clean_ans)

    ans_vocab = list(set(ans_vocab))
    ans_vocab.sort()
    ans_vocab.insert(0, "<unk>")

    with open("./data/ans_vocab.json", "w") as f:
        json.dump(ans_vocab, f, indent=4)
    print(f"Saved {len(ans_vocab)} answer tokens.")


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    build_vocab(dataset)


if __name__ == "__main__":
    main()
