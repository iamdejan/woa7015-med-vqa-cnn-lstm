from datasets import load_dataset
import re
import json


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
        if vocab in self.vocab2idx:
            return self.vocab2idx[vocab]
        else:
            return self.vocab2idx['<unk>']

    def idx2word(self, idx):
        return self.vocab[idx]



# regex for word
REGEX = re.compile(r'(\W+)')


def build_question_vocab(dataset, set_name: str):
    # build question vocabulary
    q_vocab = []
    for question in dataset["question"]:
        split = REGEX.split(question.lower())
        tmp = [w.strip() for w in split if len(w.strip()) > 0]
        q_vocab.extend(tmp)

    q_vocab = list(set(q_vocab))
    q_vocab.sort()
    q_vocab.insert(0, '<pad>')
    q_vocab.insert(1, '<unk>')
    with open(f"./data/{set_name}/q_vocab.json", 'w') as json_file:
        json.dump(q_vocab, json_file, indent=4)


def build_answer_vocab(dataset, set_name: str):
    ans_vocab = []
    for answer in dataset["answer"]:
        split = REGEX.split(answer.lower())
        tmp = [w.strip() for w in split if len(w.strip()) > 0 and not re.search(r'[^\w\s]', w.strip())]
        ans_vocab.extend(tmp)
    ans_vocab = list(set(ans_vocab))
    ans_vocab.sort()
    ans_vocab.insert(0, '<unk>')
    with open(f"./data/{set_name}/ans_vocab.json", 'w') as json_file:
        json.dump(ans_vocab, json_file, indent=4)


SETS = ["train", "test"]


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir='./cache')
    
    for set_name in SETS:
        set = dataset["test"]

        build_question_vocab(set, set_name)
        build_answer_vocab(set, set_name)


if __name__ == "__main__":
    main()
