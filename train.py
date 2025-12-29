import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from torch import optim

import utils
from build_vocab import Vocab
from model import VQAModel

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-4  # Decrease from 0.001 to 0.0001
STEP_SIZE = 10
GAMMA = 0.1
EPOCHS = 500
TRAIN = r"train"
VAL = r"val"
LOG_DIR = r"./log"
CHECKPOINT_DIR = r"./checkpoint"


def main():
    # create directories for log and checkpoint
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)

    dataset = load_dataset("flaviagiammarino/vqa-rad", cache_dir="./cache")
    train_dataset = dataset[TRAIN]
    new_split_dataset = train_dataset.train_test_split(test_size=0.2, shuffle=True)
    train_dataset = new_split_dataset[TRAIN]
    val_dataset = new_split_dataset["test"]
    new_split_dataset[VAL] = val_dataset
    del new_split_dataset["test"]

    question_vocab = Vocab("./data/q_vocab.json")
    answer_vocab = Vocab("./data/ans_vocab.json")

    model = VQAModel(
        utils.FEATURE_SIZE,
        question_vocab.vocab_size,
        answer_vocab.vocab_size,
        utils.WORD_EMBED,
        utils.HIDDEN_SIZE,
        utils.NUM_HIDDEN,
    )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    print(">> start training")
    start_time = time.time()
    for epoch in range(EPOCHS):
        epoch_loss = {key: 0 for key in [TRAIN, VAL]}

        model.train()
        for item in train_dataset:
            optimizer.zero_grad()

            image_tensor = utils.image_to_tensor(item, device)
            question_tensor = utils.question_string_to_tensor(item, question_vocab, device)
            answer_tensor = utils.answer_string_to_tensor(item, answer_vocab, device)

            # forward
            logits = model(image_tensor, question_tensor)
            loss = criterion(logits, answer_tensor)
            epoch_loss[TRAIN] += loss.item()
            # backward
            loss.backward()
            optimizer.step()

        model.eval()
        for item in val_dataset:
            image_tensor = utils.image_to_tensor(item, device)
            question_tensor = utils.question_string_to_tensor(item, question_vocab, device)
            answer_tensor = utils.answer_string_to_tensor(item, answer_vocab, device)

            with torch.no_grad():
                logits = model(image_tensor, question_tensor)
                loss = criterion(logits, answer_tensor)
            epoch_loss[VAL] += loss.item()

        # statistic
        for phase in [TRAIN, VAL]:
            epoch_loss[phase] /= len(new_split_dataset[phase])
            with open(os.path.join(LOG_DIR, f"{phase}_log.txt"), "a") as f:
                f.write(str(epoch + 1) + "\t" + str(epoch_loss[phase]) + "\n")
        print("Epoch:{}/{} | Training Loss: {train:6f} | Validation Loss: {val:6f}".format(epoch + 1, EPOCHS, **epoch_loss))

        scheduler.step()
        early_stop = early_stopping(model, epoch_loss[VAL], patience=10)
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"model-epoch-{epoch + 1}.pt"))
        if early_stop:
            print(f">> Early stop at {epoch + 1} epoch")
            break

    end_time = time.time()
    training_time = end_time - start_time
    print(f">> Finishing training | Training Time:{training_time // 60:.0f}m:{training_time % 60:.0f}s")


def early_stopping(model, epoch_loss, patience=7):
    early_stop = False
    if not bool(early_stopping.__dict__):
        early_stopping.best_loss = 1e9 + 7
        early_stopping.record_loss = 1e9 + 7
        early_stopping.counter = 0

    if epoch_loss < early_stopping.best_loss:
        early_stopping.best_loss = epoch_loss
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    if epoch_loss > early_stopping.record_loss:
        early_stopping.counter += 1
        if early_stopping.counter >= patience:
            early_stop = True
    else:
        early_stopping.counter = 0
        early_stopping.record_loss = epoch_loss

    return early_stop


if __name__ == "__main__":
    main()
