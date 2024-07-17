import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from PIL import Image
import pandas as pd
from collections import Counter
from statistics import mode
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def process_text(text):
    # 句読点の削除
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 単語に分割
    words = text.split()

    # ストップワードの削除
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

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
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import numpy as np

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
num_epoch = 11
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
    
# テストデータのロード
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./valid", transform=transform, answer=False)

# 訓練データを使用して辞書を更新する
test_dataset.update_dict(train_dataset)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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