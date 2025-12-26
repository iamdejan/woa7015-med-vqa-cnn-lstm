import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisionModel(nn.Module):
    def __init__(self, embed_dim):
        super(VisionModel, self).__init__()
        # 1. Load Pretrained Model
        self.model = models.vgg19(pretrained=True)

        # 2. FREEZE WEIGHTS (Crucial Step)
        for param in self.model.parameters():
            param.requires_grad = False

        # 3. Modify Classifier
        # VGG19 classifier has structure: Linear-ReLU-Dropout-Linear-ReLU-Dropout-Linear
        # We keep the feature extraction part but replace the very last layer
        in_features = self.model.classifier[-1].in_features  # 4096

        # Remove the last layer (class scores)
        self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])

        # Add our own trainable layer
        self.fc = nn.Linear(in_features, embed_dim)

    def forward(self, image):
        # We don't need no_grad() context manager here if we set requires_grad=False above,
        # but it doesn't hurt.
        with torch.no_grad():
            img_feature = self.model(image)

        # The new FC layer IS trainable, so it happens outside no_grad
        img_feature = self.fc(img_feature)

        l2_norm = F.normalize(img_feature, p=2, dim=1)
        return l2_norm


class QuestionModel(nn.Module):
    def __init__(self, qu_vocab_size, word_embed, hidden_size, num_hidden, qu_feature_size):
        super(QuestionModel, self).__init__()
        self.word_embedding = nn.Embedding(qu_vocab_size, word_embed)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(word_embed, hidden_size, num_hidden)  # input_feature, hidden_feature, num_layer
        self.fc = nn.Linear(2 * num_hidden * hidden_size, qu_feature_size)

    def forward(self, question):
        qu_embedding = self.word_embedding(question)  # (batchsize, qu_length=30, word_embed=300)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)  # (qu_length=30, batchsize, word_embed=300)
        _, (hidden, cell) = self.lstm(qu_embedding)  # (num_layer=2, batchsize, hidden_size=1024)
        qu_feature = torch.cat((hidden, cell), dim=2)  # (num_layer=2, batchsize, 2*hidden_size=1024)
        qu_feature = qu_feature.transpose(0, 1)  # (batchsize, num_layer=2, 2*hidden_size=1024)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)  # (batchsize, 2*num_layer*hidden_size=2048)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)  # (batchsize, qu_feature_size=1024)
        return qu_feature


class VQAModel(nn.Module):
    def __init__(self, feature_size, qu_vocab_size, ans_vocab_size, word_embed, hidden_size, num_hidden):
        super(VQAModel, self).__init__()
        self.img_encoder = VisionModel(feature_size)
        self.qu_encoder = QuestionModel(qu_vocab_size, word_embed, hidden_size, num_hidden, feature_size)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(feature_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, image, question):
        img_feature = self.img_encoder(image)  # (batchsize, feature_size=1024)
        qst_feature = self.qu_encoder(question)
        combined_feature = img_feature * qst_feature
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.tanh(combined_feature)
        combined_feature = self.fc1(combined_feature)  # (batchsize, ans_vocab_size=1000)
        combined_feature = self.dropout(combined_feature)
        combined_feature = self.tanh(combined_feature)
        logits = self.fc2(combined_feature)  # (batchsize, ans_vocab_size=1000)
        return logits
