import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VisionModel(nn.Module):
    def __init__(self, embed_dim):
        super(VisionModel, self).__init__()
        vgg = models.vgg19(pretrained=True)

        # Take the "Features" (Convolutional layers), NOT the classifier
        # Output shape: (Batch, 512, 7, 7)
        self.features = vgg.features

        # Freeze weights
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, image):
        with torch.no_grad():
            # Output: [Batch, 512, 7, 7]
            features = self.features(image)
        return features


class Attention(nn.Module):
    def __init__(self, v_dim, q_dim, mid_dim):
        super(Attention, self).__init__()
        self.v_proj = nn.Linear(v_dim, mid_dim)
        self.q_proj = nn.Linear(q_dim, mid_dim)
        self.attn = nn.Linear(mid_dim, 1)

    def forward(self, v, q):
        # v: [Batch, 49, 512] (Image regions)
        # q: [Batch, 1024]    (Question feature)

        # Expand q to match v: [Batch, 49, 1024]
        q_expanded = q.unsqueeze(1).repeat(1, v.size(1), 1)

        # Calculate scores
        # "Is this image region relevant to this question?"
        score = torch.tanh(self.v_proj(v) + self.q_proj(q_expanded))
        score = self.attn(score)  # [Batch, 49, 1]

        # Softmax to get probabilities (weights)
        weights = F.softmax(score, dim=1)

        # Weighted sum of image regions
        # [Batch, 49, 1] * [Batch, 49, 512] -> Sum -> [Batch, 512]
        attended_v = (weights * v).sum(dim=1)

        return attended_v


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

        # Attention Layer
        # v_dim=512 (VGG output), q_dim=1024 (LSTM output)
        self.attention = Attention(v_dim=512, q_dim=feature_size, mid_dim=512)

        self.fc1 = nn.Linear(512 + feature_size, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, image, question):
        # 1. Get Image Features: [Batch, 512, 7, 7]
        img_feat = self.img_encoder(image)

        # Flatten to [Batch, 49, 512] (49 regions)
        batch, ch, h, w = img_feat.size()
        img_feat = img_feat.view(batch, ch, -1).permute(0, 2, 1)

        # 2. Get Question Features: [Batch, 1024]
        qst_feat = self.qu_encoder(question)

        # 3. Apply Attention
        # "Focus on the region described by the question"
        attended_img = self.attention(img_feat, qst_feat)  # [Batch, 512]

        # 4. Fuse and Predict
        combined = torch.cat((attended_img, qst_feat), dim=1)
        out = self.fc1(combined)
        return out
