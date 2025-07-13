import os
import sys
import cv2
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold
'''This Script is for running it multiple times and saving the metics in csv files.'''
# ==================== Log Output to File ====================
class TeeOutput:
    def __init__(self, filename):
        self.file = open(filename, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

sys.stdout = TeeOutput("vgg16_output_log.txt")
sys.stderr = sys.stdout

# ==================== Load and preprocess data ====================
target_shape = (224, 224)
metadata = pd.read_csv('spectrogram_metadata.csv')
label_map = {'normal': 0, 'murmur': 1}
X, Y = [], []

def resize_spec(spec):
    return cv2.resize(spec, target_shape, interpolation=cv2.INTER_AREA)

for idx, row in metadata.iterrows():
    path = os.path.join('spectrograms_png', row['png'])
    spec = cv2.imread(path, cv2.IMREAD_COLOR)
    if spec is None:
        print(f"Warning: Failed to load image at {path}")
        continue
    spec = cv2.cvtColor(spec, cv2.COLOR_BGR2RGB)
    spec = resize_spec(spec).astype(np.float32) / 255.0
    X.append(spec)
    Y.append(label_map[row['label']])

X = np.array(X).transpose(0, 3, 1, 2)
Y = np.array(Y)

# ==================== Dataset & Dataloader ====================
imagenet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])

class SpectrogramDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        return x, self.Y[idx]

# ==================== Model Setup ====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_vgg16():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    conv_count = 0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            if conv_count <= 5:
                for param in layer.parameters():
                    param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 2)
    return model.to(device)

# ==================== Train One Fold ====================
def train(training_X, training_Y, validation_X, validation_Y, fold, epochs, patience=5, lr=1e-4):
    print(f"\n---------------------------\nTraining Fold {fold}...\n---------------------------")
    model = initialize_vgg16()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    train_dataset = SpectrogramDataset(training_X, training_Y)
    val_dataset = SpectrogramDataset(validation_X, validation_Y)

    class_counts = np.bincount(training_Y.astype(int))
    class_weights = 1. / class_counts
    sample_weights = class_weights[training_Y.astype(int)]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16)

    best_model = None
    best_f1 = 0
    trigger_times = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X_train, y_train in train_loader:
            X_train, y_train = X_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        y_true, y_pred = [], []
        total_val_loss = 0.0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output, y_val)
                total_val_loss += loss.item()
                pred = output.argmax(dim=1).cpu().numpy()
                y_true.extend(y_val.cpu().numpy())
                y_pred.extend(pred)

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"[Fold {fold}]Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val loss: {avg_val_loss:.4f} | F1: {f1:.4f}")

        if epoch == 0 or avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = copy.deepcopy(model.state_dict())
            os.makedirs("weights", exist_ok=True)
            torch.save(best_model, os.path.join("weights", f"model_fold_{fold}.pt"))
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss.")
                break

    model.load_state_dict(best_model)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            pred = output.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred)

    acc_final = accuracy_score(y_true, y_pred)
    f1_final = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n---------------------------")
    print(f"Metrics Fold {fold}...")
    print("---------------------------")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"Validation Accuracy: {acc_final:.4f}")
    print(f"Validation F1 Score: {f1_final:.4f}")
    print(f"Precision: {precision:.4f}")
    print("---------------------------\n")

    return {
        'fold': fold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': acc_final,
        'f1': f1_final,
        'precision': precision
    }

# ==================== Run K-Fold for One Run ====================
def run_kfold(X, Y, run_id, epochs, patience, lr, n_splits):
    print(f"\n========= RUN {run_id + 1}/10 =========")
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=run_id)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        result = train(X_train, y_train, X_val, y_val, fold, epochs, patience, lr)
        result['run'] = run_id + 1
        fold_metrics.append(result)

    avg_f1 = np.mean([m['f1'] for m in fold_metrics])
    avg_acc = np.mean([m['accuracy'] for m in fold_metrics])

    print("\n==== Final Cross-Validation Report ====")
    print("-------------------------------------------------------")
    print(f"  Average F1 over all folds:           {avg_f1:.4f}")
    print(f"  Average accuracy over all folds:     {avg_acc:.4f}")
    print("-------------------------------------------------------")

    return {
        'run': run_id + 1,
        'avg_f1': avg_f1,
        'avg_acc': avg_acc,
        'folds': fold_metrics
    }

# ==================== Main: 10 Runs ====================
def main():
    num_runs = 10
    epochs = 20
    n_splits = 5
    lr = 0.00005
    patience = 3

    all_runs_metrics = []

    for run in range(num_runs):
        result = run_kfold(X, Y, run_id=run, epochs=epochs, patience=patience, lr=lr, n_splits=n_splits)
        all_runs_metrics.append(result)

    # ==================== Save Metrics ====================
    run_level_df = pd.DataFrame([{
        'run': r['run'],
        'avg_f1': r['avg_f1'],
        'avg_accuracy': r['avg_acc']
    } for r in all_runs_metrics])

    fold_level_df = pd.DataFrame([
        {**fold, 'run': run['run']} for run in all_runs_metrics for fold in run['folds']
    ])

    run_level_df.to_csv("vgg16_run_level_metrics.csv", index=False)
    fold_level_df.to_csv("vgg16_fold_level_metrics.csv", index=False)

    overall_f1 = fold_level_df['f1'].mean()
    overall_acc = fold_level_df['accuracy'].mean()
    print("\n-------- Overall Performance Across All Runs & Folds ---------")
    print(f"Average F1:  {overall_f1:.4f}")
    print(f"Average Acc: {overall_acc:.4f}")
    print("---------------------------------------------------------------")

    best_idx = fold_level_df['f1'].idxmax()
    best_model = fold_level_df.loc[best_idx]
    print(f"Best overall model: Run {best_model['run']} | Fold {best_model['fold']}")
    print(f"F1: {best_model['f1']:.4f} | Accuracy: {best_model['accuracy']:.4f}")

    print("Metrics saved in CSV files.")

if __name__ == '__main__':
    main()