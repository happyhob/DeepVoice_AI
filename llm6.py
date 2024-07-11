#epochs 수 변경

###config 값 수정

'''

55384개의 음성 데이터의 크기(67KB 최대, 9KB 최소, 평균 30KB)를 고려한 Config 설정은 다음과 같이 할 수 있습니다. 
여기서는 데이터 크기에 기반하여 배치 사이즈와 학습률 등을 조정합니다. 
또한, 음성 데이터의 길이를 일정하게 맞추기 위해 패딩을 적용합니다.
'''


'''
주요 설정 이유
MAX_LEN: 300으로 설정하여 평균 길이에 맞추어 음성 데이터를 패딩합니다.
너무 짧으면 정보 손실이 발생하고, 너무 길면 패딩이 과도하게 추가될 수 있습니다.
BATCH_SIZE: 128로 설정하지만, 사용 중인 GPU 메모리에 맞추어 더 작거나 크게 조정할 수 있습니다.
N_EPOCHS: 10으로 설정하여 충분한 학습을 진행하면서도 과적합을 방지합니다.
LR (Learning Rate): 1e-4로 설정하여 안정적인 학습을 유도합니다. 학습 진행 중 학습률 스케줄러를 사용하여 조정할 수 있습니다.
'''
import librosa
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch
import torchmetrics
import os
import warnings
from torch.optim.lr_scheduler import ReduceLROnPlateau



warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class Config:
    SR = 32000
    N_MFCC = 13
    MAX_LEN = 300  # 음성 데이터의 평균 길이에 맞추어 설정
    # Dataset
    ROOT_FOLDER = './'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 128  # GPU 메모리에 따라 조정 (예: 128)
    N_EPOCHS = 20  # 과적합을 방지하기 위해 적절한 에포크 수 설정
    LR = 1e-4  # 학습률 설정
    # Othersㄴ
    SEED = 42

    
CONFIG = Config()


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CONFIG.SEED)  # Seed 고정

df = pd.read_csv('./train.csv')
train, val, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)

'''
데이터 전처리 수정
RCNN 모델을 사용하기 위해 데이터 전처리 부분에서 MFCC 특징을 추출할 때 패딩을 적용합니다.
'''
def get_mfcc_feature(df, train_mode=True):
    features = []
    labels = []
    for _, row in tqdm(df.iterrows()):
        y, sr = librosa.load(row['path'], sr=CONFIG.SR)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=CONFIG.N_MFCC)
        if mfcc.shape[1] < CONFIG.MAX_LEN:
            pad_width = CONFIG.MAX_LEN - mfcc.shape[1]
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :CONFIG.MAX_LEN]
        features.append(mfcc)

        if train_mode:
            label = row['label']
            label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
            label_vector[0 if label == 'fake' else 1] = 1
            labels.append(label_vector)

    if train_mode:
        return features, labels
    return features


train_mfcc, train_labels = get_mfcc_feature(train, True)
val_mfcc, val_labels = get_mfcc_feature(val, True)

class CustomDataset(Dataset):
    def __init__(self, mfcc, label):
        self.mfcc = mfcc
        self.label = label

    def __len__(self):
        return len(self.mfcc)

    def __getitem__(self, index):
        mfcc = self.mfcc[index]
        if self.label is not None:
            return mfcc, self.label[index]
        return mfcc

train_dataset = CustomDataset(train_mfcc, train_labels)
val_dataset = CustomDataset(val_mfcc, val_labels)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss



'''
RCNN 모델 설계
RNN과 CNN의 조합을 사용한 RCNN 모델을 설계합니다. CNN을 통해 특징을 추출하고, RNN을 통해 시계열 정보를 학습합니다.
'''
class RCNN(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES, num_layers=2):
        super(RCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.rnn = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, num_features)
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Take the last time step
        x = self.sigmoid(x)
        return x


from sklearn.metrics import roc_auc_score

from torch.optim.lr_scheduler import ReduceLROnPlateau

'''
학습 루프 수정
학습 루프에서 학습률 스케줄러를 추가하여 검증 손실이 개선되지 않을 때 학습률을 줄입니다.
'''
def train(model, optimizer, train_loader, val_loader, device, n_epochs=50, patience=5):
    model.to(device)
    criterion = nn.BCELoss().to(device)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    best_val_score = 0
    best_model = None
    
    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss = []
        for features, labels in tqdm(iter(train_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            optimizer.zero_grad()
            
            output = model(features)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        _val_loss, _val_score = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Validation AUC : [{_val_score:.5f}]')
            
        scheduler.step(_val_loss)
        early_stopping(_val_loss, model)
            
        if best_val_score < _val_score:
            best_val_score = _val_score
            best_model = model

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.load_state_dict(torch.load('checkpoint.pt'))
    return model



def multiLabel_AUC(y_true, y_scores):
    auc_scores = []
    for i in range(y_true.shape[1]):
        auc = roc_auc_score(y_true[:, i], y_scores[:, i])
        auc_scores.append(auc)
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score
    
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss, all_labels, all_probs = [], [], []
    
    with torch.no_grad():
        for features, labels in tqdm(iter(val_loader)):
            features = features.float().to(device)
            labels = labels.float().to(device)
            
            probs = model(features)
            
            loss = criterion(probs, labels)

            val_loss.append(loss.item())

            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
        
        _val_loss = np.mean(val_loss)

        all_labels = np.concatenate(all_labels, axis=0)
        all_probs = np.concatenate(all_probs, axis=0)
        
        # Calculate AUC score
        auc_score = multiLabel_AUC(all_labels, all_probs)
    
    return _val_loss, auc_score

model = RCNN()
optimizer = torch.optim.Adam(params=model.parameters(), lr=CONFIG.LR)

infer_model = train(model, optimizer, train_loader, val_loader, device, n_epochs=50, patience=5)


test = pd.read_csv('./test.csv')
test_mfcc = get_mfcc_feature(test, False)
test_dataset = CustomDataset(test_mfcc, None)
test_loader = DataLoader(
    test_dataset,
    batch_size=CONFIG.BATCH_SIZE,
    shuffle=False
)

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(iter(test_loader)):
            features = features.float().to(device)
            
            probs = model(features)

            probs  = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

preds = inference(infer_model, test_loader, device)

submit = pd.read_csv('./sample_submission.csv')
submit.iloc[:, 1:] = preds
submit.head()

submit.to_csv('./rcnn_submit3.csv', index=False)
