import torchaudio
from sklearn.model_selection import train_test_split, StratifiedKFold
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

warnings.filterwarnings('ignore')  # 경고 메시지 무시

# GPU 사용 가능 여부 확인
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Config 클래스 설정
class Config:
    SR = 32000  # 샘플링 레이트
    N_MFCC = 13  # MFCC 계수 개수
    ROOT_FOLDER = './'  # 데이터 루트 폴더 경로
    N_CLASSES = 2  # 클래스 개수
    BATCH_SIZE = 96  # 배치 사이즈
    N_EPOCHS = 200  # 에폭 개수
    LR = 3e-4  # 학습률
    SEED = 42  # 랜덤 시드
    N_SPLITS = 5  # 교차 검증 폴드 수

CONFIG = Config()  # 설정 객체 생성

# 랜덤 시드 설정 함수
def seed_everything(seed):
    random.seed(seed)  # Python 랜덤 시드 설정
    os.environ['PYTHONHASHSEED'] = str(seed)  # Python 해시 시드 설정
    np.random.seed(seed)  # NumPy 랜덤 시드 설정
    torch.manual_seed(seed)  # PyTorch 랜덤 시드 설정
    torch.cuda.manual_seed(seed)  # GPU 랜덤 시드 설정
    torch.backends.cudnn.deterministic = True  # CuDNN 결정적 설정
    torch.backends.cudnn.benchmark = True  # CuDNN 벤치마크 설정

seed_everything(CONFIG.SEED)  # 시드 설정

# MFCC 특징 추출 함수
def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = torchaudio.transforms.MFCC(
        sample_rate=sr, 
        n_mfcc=n_mfcc, 
        melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": 23, "center": False}
    )(torch.tensor(y))  # MFCC 변환
    mfcc = mfcc.mean(dim=1).numpy()  # MFCC 평균값 계산
    mfcc = (mfcc - np.mean(mfcc)) / np.std(mfcc)  # 정규화
    return mfcc  # MFCC 반환

# 데이터 로드 및 특징 추출 함수
def load_and_extract(row):
    waveform, sr = torchaudio.load(row['path'])  # 오디오 파일 로드
    waveform = waveform.numpy().flatten()  # 1차원 배열로 변환
    mfcc = extract_mfcc(waveform, sr, n_mfcc=CONFIG.N_MFCC)  # MFCC 특징 추출
    if 'label' in row:
        label = row['label']  # 라벨 추출
        label_vector = np.zeros(CONFIG.N_CLASSES, dtype=float)
        label_vector[0 if label == 'fake' else 1] = 1  # 라벨 벡터화
        return mfcc, label_vector  # 특징과 라벨 반환
    return mfcc  # 특징 반환

# 특징 및 라벨 얻기 함수
def get_features_and_labels(df, train_mode=True):
    with ProcessPoolExecutor() as executor:
        if train_mode:
            results = list(tqdm(executor.map(load_and_extract, df.to_dict('records')), total=len(df)))  # 병렬 처리
            features, labels = zip(*results)  # 특징과 라벨 분리
            return list(features), list(labels)  # 리스트로 변환
        else:
            features = list(tqdm(executor.map(load_and_extract, df.to_dict('records')), total=len(df)))  # 병렬 처리
            return features  # 특징 반환

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, features, labels=None, transform=None):
        self.features = features  # 특징 저장
        self.labels = labels  # 라벨 저장
        self.transform = transform  # 변환 저장

    def __len__(self):
        return len(self.features)  # 데이터셋 크기 반환

    def __getitem__(self, index):
        feature = self.features[index]  # 인덱스로 특징 가져오기
        if self.transform:
            feature = self.transform(feature)  # 변환 적용
        if self.labels is not None:
            label = self.labels[index]  # 인덱스로 라벨 가져오기
            return feature, label  # 특징과 라벨 반환
        return feature  # 특징 반환

# MLP 모델 정의
class MLP(nn.Module):
    def __init__(self, input_dim=CONFIG.N_MFCC, hidden_dim=64, output_dim=CONFIG.N_CLASSES):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 첫 번째 전결합층
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # 두 번째 전결합층
        self.fc3 = nn.Linear(hidden_dim, output_dim)  # 세 번째 전결합층
        self.relu = nn.ReLU()  # ReLU 활성화 함수
        self.dropout = nn.Dropout(0.5)  # 드롭아웃

    def forward(self, x):
        x = self.relu(self.fc1(x))  # 첫 번째 레이어 통과
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.relu(self.fc2(x))  # 두 번째 레이어 통과
        x = self.dropout(x)  # 드롭아웃 적용
        x = self.fc3(x)  # 세 번째 레이어 통과
        return x  # 출력 반환

# AUC 계산 함수
def calculate_auc(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)  # AUC 계산
    return auc  # AUC 반환

# 모델 학습 함수
def train(model, optimizer, train_loader, val_loader, device):
    model.to(device)  # 모델을 디바이스로 이동
    criterion = nn.BCEWithLogitsLoss().to(device)  # 손실 함수 설정
    scaler = GradScaler()  # Mixed Precision을 위한 스케일러 설정
    best_val_auc = 0  # 최고 검증 AUC 초기화
    best_model = None  # 최고 모델 초기화
    patience = 10  # 조기 종료를 위한 patience
    wait = 0  # 조기 종료를 위한 대기 변수

    for epoch in range(1, CONFIG.N_EPOCHS + 1):
        model.train()  # 모델을 학습 모드로 설정
        train_losses = []  # 학습 손실 리스트 초기화
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch}/{CONFIG.N_EPOCHS}', leave=False):
            features = features.to(device).float()  # 특징을 디바이스로 이동
            labels = labels.to(device).float()  # 라벨을 디바이스로 이동
            optimizer.zero_grad()  # 옵티마이저 초기화
            with autocast():
                outputs = model(features)  # 모델 예측
                loss = criterion(outputs, labels)  # 손실 계산
            scaler.scale(loss).backward()  # 역전파
            scaler.step(optimizer)  # 옵티마이저 스텝
            scaler.update()  # 스케일러 업데이트
            train_losses.append(loss.item())  # 손실 추가
        avg_train_loss = np.mean(train_losses)  # 평균 학습 손실 계산
        val_loss, val_auc = validation(model, criterion, val_loader, device)  # 검증
        print(f'Epoch [{epoch}/{CONFIG.N_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc  # 최고 검증 AUC 업데이트
            best_model = model  # 최고 모델 업데이트
            wait = 0  # 대기 변수 초기화
        else:
            wait += 1  # 대기 변수 증가
        if wait >= patience:
            print(f'Early stopping at epoch {epoch}')
            break  # 조기 종료
    return best_model  # 최고 모델 반환

# 다중 라벨 AUC 계산 함수
def multiLabel_AUC(y_true, y_scores):
    auc_scores = [roc_auc_score(y_true[:, i], y_scores[:, i]) for i in range(y_true.shape[1])]  # 각 클래스별 AUC 계산
    mean_auc_score = np.mean(auc_scores)  # 평균 AUC 계산
    return mean_auc_score  # 평균 AUC 반환

# 검증 함수
def validation(model, criterion, val_loader, device):
    model.eval()  # 모델을 평가 모드로 설정
    val_losses = []  # 검증 손실 리스트 초기화
    all_labels = []  # 모든 라벨 리스트 초기화
    all_preds = []  # 모든 예측 리스트 초기화
    with torch.no_grad():  # 그라디언트 계산 비활성화
        for features, labels in tqdm(val_loader, desc='Validating', leave=False):
            features = features.to(device)  # 특징을 디바이스로 이동
            labels = labels.to(device).float()  # 라벨을 디바이스로 이동
            outputs = model(features)  # 모델 예측
            loss = criterion(outputs, labels)  # 손실 계산
            val_losses.append(loss.item())  # 손실 추가
            all_labels.extend(labels.cpu().detach().numpy())  # 라벨 추가
            all_preds.extend(outputs.cpu().detach().numpy())  # 예측 추가
    avg_val_loss = np.mean(val_losses)  # 평균 검증 손실 계산
    val_auc = multiLabel_AUC(np.array(all_labels), np.array(all_preds))  # AUC 계산
    return avg_val_loss, val_auc  # 검증 손실과 AUC 반환

# 추론 함수
def inference(model, test_loader, device):
    model.to(device)  # 모델을 디바이스로 이동
    model.eval()  # 모델을 평가 모드로 설정
    predictions = []  # 예측 리스트 초기화
    with torch.no_grad():  # 그라디언트 계산 비활성화
        for features in tqdm(test_loader, desc='Inferencing', leave=False):
            features = features.float().to(device)  # 특징을 디바이스로 이동
            with autocast():
                probs = model(features)  # 모델 예측
            probs = torch.sigmoid(probs)  # 시그모이드 활성화 함수 적용
            probs = probs.cpu().detach().numpy()  # 예측을 CPU로 이동
            predictions += probs.tolist()  # 예측 추가
    return predictions  # 예측 반환

# 메인 함수
def main():
    seed_everything(CONFIG.SEED)  # 시드 설정
    df = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'train.csv'))  # 학습 데이터 로드
    skf = StratifiedKFold(n_splits=CONFIG.N_SPLITS, shuffle=True, random_state=CONFIG.SEED)  # 교차 검증 설정
    
    test = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'test.csv'))  # 테스트 데이터 로드
    test_features = get_features_and_labels(test, train_mode=False)  # 테스트 특징 추출
    test_dataset = CustomDataset(test_features)  # 테스트 데이터셋 생성
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)  # 테스트 데이터 로더 생성

    all_predictions = np.zeros((len(test), CONFIG.N_CLASSES))  # 모든 예측 초기화

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f'Fold {fold+1}/{CONFIG.N_SPLITS}')
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]  # 학습 및 검증 데이터 분할
        train_features, train_labels = get_features_and_labels(train_df, train_mode=True)  # 학습 특징 추출
        val_features, val_labels = get_features_and_labels(val_df, train_mode=True)  # 검증 특징 추출
        train_dataset = CustomDataset(train_features, train_labels)  # 학습 데이터셋 생성
        val_dataset = CustomDataset(val_features, val_labels)  # 검증 데이터셋 생성
        train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=4)  # 학습 데이터 로더 생성
        val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False, num_workers=4)  # 검증 데이터 로더 생성
        model = MLP()  # MLP 모델 생성
        optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG.LR)  # 옵티마이저 설정
        best_model = train(model, optimizer, train_loader, val_loader, device)  # 모델 학습
        
        fold_predictions = inference(best_model, test_loader, device)  # 추론
        all_predictions += np.array(fold_predictions)  # 예측 합산

    all_predictions /= CONFIG.N_SPLITS  # 평균 예측 계산

    submit = pd.read_csv(os.path.join(CONFIG.ROOT_FOLDER, 'sample_submission.csv'))  # 제출 파일 로드
    submit.iloc[:, 1:] = all_predictions  # 예측 삽입
    submit.to_csv(os.path.join(CONFIG.ROOT_FOLDER, 'just7.csv'), index=False)  # 제출 파일 저장

if __name__ == "__main__":
    main()  # 메인 함수 실행
