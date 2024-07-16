import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
from collections import Counter
from statistics import mode

def process_text(text):
    # テキストの前処理を行う関数（例：小文字化、不要な記号の削除など）
    return text.lower()

class VQADataset(Dataset):
    def __init__(self, df_path, image_dir, transform=None, answer=True):
        self.transform = transform
        self.image_dir = image_dir
        self.df = pd.read_json(df_path)
        self.answer = answer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        if self.answer:
            self.answer2idx = {}
            for answers in self.df["answers"]:
                for answer in answers:
                    word = process_text(answer["answer"])
                    if word not in self.answer2idx:
                        self.answer2idx[word] = len(self.answer2idx)
            self.idx2answer = {v: k for k, v in self.answer2idx.items()}

    def update_dict(self, dataset):
        self.answer2idx = dataset.answer2idx
        self.idx2answer = dataset.idx2answer

    def __getitem__(self, idx):
        image = Image.open(f"{self.image_dir}/{self.df['image'][idx]}")
        if self.transform:
            image = self.transform(image)

        question = process_text(self.df["question"][idx])
        encoding = self.tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        if self.answer:
            answers = [self.answer2idx[process_text(answer["answer"])] for answer in self.df["answers"][idx]]
            mode_answer_idx = mode(answers)

            return image, input_ids, attention_mask, torch.tensor(answers), torch.tensor(int(mode_answer_idx))

        else:
            return image, input_ids, attention_mask

    def __len__(self):
        return len(self.df)

import torch
import torch.nn as nn
import torchvision.models as models
from transformers import BertModel

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super(VQAModel, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.vgg16 = nn.Sequential(*list(vgg16.features.children()), nn.Flatten())
        self.vgg16_output_dim = 25088  # VGG16の出力特徴量の次元数

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(self.vgg16_output_dim + 512 * 2, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, input_ids, attention_mask):
        image_feature = self.vgg16(image)  # 画像の特徴量
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        question_feature, _ = self.lstm(bert_output.last_hidden_state)  # BERTの出力をLSTMに入力
        question_feature = question_feature[:, -1, :]  # 最後のLSTMの出力を使用

        x = torch.cat([image_feature, question_feature], dim=1)
        x = self.fc(x)

        return x

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# データセットの準備
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = VQADataset(df_path='./data/train.json', image_dir='./train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# モデル、損失関数、オプティマイザの準備
model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練ループ
num_epoch = 15
for epoch in range(num_epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    optimizer.zero_grad()

    for i, (image, input_ids, attention_mask, answers, mode_answer_idx) in enumerate(train_loader):
        image, input_ids, attention_mask, mode_answer_idx = image.to(device), input_ids.to(device), attention_mask.to(device), mode_answer_idx.to(device)
        output = model(image, input_ids, attention_mask)
        loss = criterion(output, mode_answer_idx)
        loss.backward()

        if (i + 1) % 2 == 0:  # accumulation_steps = 4
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == mode_answer_idx).sum().item()
        total += mode_answer_idx.size(0)
        if (i%500==0):
            print(i,'/',len(train_loader))

    end_time = time.time()
    train_loss = total_loss / total
    train_acc = correct / total
    train_time = end_time - start_time

    print(f"【{epoch + 1}/{num_epoch}】\n"
          f"train time: {train_time:.2f} [s]\n"
          f"train loss: {train_loss:.4f}\n"
          f"train acc: {train_acc:.4f}")
    
# 提出用ファイルの作成
model.eval()
submission = []
for image, input_ids, attention_mask in test_loader:
    image = image.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    with torch.no_grad():
        pred = model(image, input_ids, attention_mask)
        pred = pred.argmax(1).cpu().item()
        submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)

# モデルの保存
torch.save(model.state_dict(), "model.pth")
# 提出ファイルの保存
np.save("submission.npy", submission)