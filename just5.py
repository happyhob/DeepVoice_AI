import torchaudio
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
from sklearn.metrics import roc_auc_score
from concurrent.futures import ProcessPoolExecutor
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class Config:
    SR = 32000
    N_MFCC = 13
    ROOT_FOLDER = './'
    N_CLASSES = 2
    BATCH_SIZE = 96
    N_EPOCHS = 200
    LR = 3e-4
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

seed_everything(CONFIG.SEED)

def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr, 
        n_mfcc=n_mfcc, 
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )(torch.tensor(y))
    mfcc = mfcc.mean(dim=1).numpy()
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)
    return mfcc

def load_and_extract(row):
    waveform, sr = torchaudio.load(row['path'])
    waveform = waveform.numpy().flatten()
    mfcc = extract_mfcc(waveform, sr, n_mfcc=CONFIG.N_MFCC)
    if 'label' in row:
        label = row['label']
        label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
        label_vector[0 if label == 'fake' else 1] = 1
        return mfcc, label_vector
    return mfcc

def get_features_and_labels(df, train_mode=True):
    with ProcessPoolExecutor() as executor:
        if train_mode:
            results = list(tqdm(executor.map(load_and_extract, df.to_dict('records')), total=len(df)))
            features, labels = zip(*results)
            return list(features), list(labels)
        else:
            features = list(tqdm(executor.map(load_and_extract, df.to_dict('records')), total=len(df)))
            return features

class CustomDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        if self.transform:
            feature = self.transform(feature)
        if self.labels is not None:
            label = self.labels[index]
            return feature, label
        return feature

class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=128, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def calculate_auc(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    return auc

def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = GradScaler()
    best_val_auc = 0
    best_model = None
    patience = 10
    wait = 0

    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()
        train_losses = []
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG.N_EPOCHS}', leave=False):
            features = features.to(device).float()
            labels = labels.to(device).float()
            optimizer.zero_grad()
            with autocast():
                outputs = model(features)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(loss.item())
        avg_train_loss = np.mean(train_losses)
        val_loss, val_auc = validation(model, criterion, val_loader, device)
        print(f'Epoch [{epoch}/{CONFIG.N_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model = model
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    return best_model

def multiLabel_AUC(y_true, y_scores):
    auc_scores = [roc_auc_score(y_true[:, i], y_scores[:, i]) for i in range(y_true.shape[1])]
    mean_auc_score = np.mean(auc_scores)
    return mean_auc_score

def validation(model, criterion, val_loader, device):
    model.eval()
    val_losses = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc='Validating', leave=False):
            features = features.to(device)
            labels = labels.to(device).float()
            outputs = model(features)
            loss = criterion(outputs, labels)
            val_losses.append(loss.item())
            all_labels.extend(labels.cpu().detach().numpy())
            all_preds.extend(outputs.cpu().detach().numpy())
    avg_val_loss = np.mean(val_losses)
    val_auc = multiLabel_AUC(np.array(all_labels), np.array(all_preds))
    return avg_val_loss, val_auc

def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for features in tqdm(test_loader, desc='Inferencing', leave=False):
            features = features.float().to(device)
            with autocast():
                probs = model(features)
            probs = torch.sigmoid(probs)  # Apply sigmoid activation here
            probs = probs.cpu().detach().numpy()
            predictions += probs.tolist()
    return predictions

def main():
    seed_everything(CONFIG.SEED)
    df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'train.csv'))
    train_df, val_df, _, _ = train_test_split(df, df['label'], test_size=0.2, random_state=CONFIG.SEED)
    train_features, train_labels = get_features_and_labels(train_df, train_mode=True)
    val_features, val_labels = get_features_and_labels(val_df, train_mode=True)
    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)
    model = MLP()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)
    best_model = train(model, optimizer, train_loader, val_loader, device)
    test = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'test.csv'))
    test_features = get_features_and_labels(test, train_mode=False)
    test_dataset = CustomDataset(test_features)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)
    predictions = inference(best_model, test_loader, device)
    submit = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'sample_submission.csv'))
    submit.iloc[:, 1:] = predictions
    submit.to_csv(os.path.join(CONFIG.ROOT_FOLDER, 'just9.csv'), index=False)

if __name__ == "__main__":
    main()
