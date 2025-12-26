import re

from datasets import load_dataset
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


def main():
    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    train_dataset = dataset["train"]
    pass


if __name__ == "__main__":
    main()
