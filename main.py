import re
import random
import time
from statistics import mode

from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from nltk.corpus import stopwords


# データフレームの読み込み
train_df = pd.read_json('./data/train.json')
val_df = pd.read_json('./data/valid.json')

# クラスマッピングの読み込み
class_mapping = pd.read_csv('https://huggingface.co/spaces/CVPR/VizWiz-CLIP-VQA/raw/main/data/annotations/class_mapping.csv')
class_mapping_dict = dict(zip(class_mapping['answer'].astype(str), class_mapping['class_id']))

# 質問文の前処理
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def process_text(question):
    question = question.lower()  # 小文字に変換
    question = re.sub(r'[^\w\s]', '', question)  # 句読点を削除
    question = ' '.join([word for word in question.split() if word not in stop_words])  # ストップワードの削除
    return question


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


from transformers import BertTokenizer


# 2. 評価指標の実装
# 簡単にするならBCEを利用する
def VQA_criterion(batch_pred: torch.Tensor, batch_answers: torch.Tensor):
    total_acc = 0.

    for pred, answers in zip(batch_pred, batch_answers):
        acc = 0.
        for i in range(len(answers)):
            num_match = 0
            for j in range(len(answers)):
                if i == j:
                    continue
                if pred == answers[j]:
                    num_match += 1
            acc += min(num_match / 3, 1)
        total_acc += acc / 10

    return total_acc / len(batch_pred)





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
from transformers import BertModel

class VQAModel(nn.Module):
    def __init__(self, n_answer: int):
        super(VQAModel, self).__init__()
        self.resnet = ResNet18()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.Linear(512 * 2 + 512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, n_answer)
        )

    def forward(self, image, input_ids, attention_mask):
        image_feature = self.resnet(image)  # 画像の特徴量
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
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# モデル、損失関数、オプティマイザの準備
model = VQAModel(n_answer=len(train_dataset.answer2idx)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 訓練ループ
num_epoch = 5
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

        if (i + 1) % 4 == 0:  # accumulation_steps = 4
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == mode_answer_idx).sum().item()
        total += mode_answer_idx.size(0)

    end_time = time.time()
    train_loss = total_loss / total
    train_acc = correct / total
    train_time = end_time - start_time

    print(f"【{epoch + 1}/{num_epoch}】\n"
          f"train time: {train_time:.2f} [s]\n"
          f"train loss: {train_loss:.4f}\n"
          f"train acc: {train_acc:.4f}")


test_dataset = VQADataset(df_path="./data/valid.json", image_dir="./valid", transform=transform, answer=False)
test_dataset.update_dict(train_dataset)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 提出用ファイルの作成
model.eval()
submission = []
for image, question in test_loader:
    image, question = image.to(device), question.to(device)
    pred = model(image, question)
    pred = pred.argmax(1).cpu().item()
    submission.append(pred)

submission = [train_dataset.idx2answer[id] for id in submission]
submission = np.array(submission)
torch.save(model.state_dict(), "model.pth")
np.save("submission.npy", submission)