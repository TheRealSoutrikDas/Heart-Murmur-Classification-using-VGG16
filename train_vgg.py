import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms as transforms
import cv2
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score
from sklearn.model_selection import StratifiedKFold
import copy

# ==================== Load and preprocess data ====================

target_shape = (224, 224) # target shape is 224 x 224 cause vgg16 takes in 224 x 224 images
metadata = pd.read_csv('spectrogram_metadata.csv')
label_map = {'normal': 0, 'murmur': 1}
X, Y = [], []

def resize_spec(spec): # resizes image to target shape[224 x 224]
    return cv2.resize(spec, target_shape, interpolation=cv2.INTER_AREA)

for idx, row in metadata.iterrows():
    path = os.path.join('spectrograms_png', row['png'])
    spec = cv2.imread(path, cv2.IMREAD_COLOR) # reads image in color[BGR]
    if spec is None:
        print(f"Warning: Failed to load image at {path}")
        continue
    spec = cv2.cvtColor(spec, cv2.COLOR_BGR2RGB) # convert color from BGR to RGB
    spec = resize_spec(spec).astype(np.float32) / 255.0 #  to 32-bit floating point and normalizes the data from a range of [0, 255] to [0.0, 1.0]
    X.append(spec)
    Y.append(label_map[row['label']])

X = np.array(X).transpose(0, 3, 1, 2)  # [N, H, W, C] to [N, C, H, W] as torch.tensor expects image in this format 
Y = np.array(Y)

# ==================== Dataset & Dataloader ====================
#defines a normalization transform used with VGG16
imagenet_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], # mean and std values are from the ImageNet dataset (on which VGG16 was trained)
                                          std=[0.229, 0.224, 0.225])
# normalizes each RGB channel of input tensor to match the distribution expected by vgg16 model: for eg Red channel: (x - 0.485) / 0.229

class SpectrogramDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32) # Convert numpy arrays to float32 tensors
        self.Y = torch.tensor(Y, dtype=torch.long) # Convert numpy arrays to long tensors (for cross entropy loss)
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x) # apply transformation
        return x, self.Y[idx]

# ==================== Model Setup ====================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_vgg16():
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1) # imports the vgg16 model with weights. for initialization without weights weights=NOne

    # Freeze first 5 conv layers
    conv_count = 0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            conv_count += 1
            if conv_count <= 5:
                for param in layer.parameters():
                    param.requires_grad = False

    # Modify classifier [for binary classification]
    model.classifier[6] = nn.Linear(4096, 2)
    return model.to(device)

# ==================== Training Setup ====================
# train function for training one fold
def train(training_X, training_Y, validation_X, validation_Y, fold, epochs, patience=5, lr=1e-4):
    print("\n---------------------------")
    print(f"Training Fold {fold}...")
    print("---------------------------\n")

    # reinitializing model for every fold and declairing loss function and optimizer
    model = initialize_vgg16()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # create training and validation dataset for the fold
    train_dataset = SpectrogramDataset(training_X, training_Y)
    val_dataset = SpectrogramDataset(validation_X, validation_Y)

    class_counts = np.bincount(training_Y.astype(int)) # count labels
    class_weights = 1. / class_counts # Compute inverse class weights ( gives a higher weight to classes with lesser labels)
    sample_weights = class_weights[training_Y.astype(int)] # Assign the weight to each sample
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights)) #Creates a WeightedRandomSampler, which will sample data points with probability proportional to their weight

    # loading data
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # for tracking best model
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
            loss = criterion(output, y_train) # Computes the loss
            loss.backward() # Backpropagates the loss for computing gradients
            optimizer.step() # Updates model weights
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
    
        model.eval()

        y_true, y_pred = [], []
        total_val_loss = 0.0

        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                output = model(X_val)
                loss = criterion(output, y_val)  # Compute val loss
                total_val_loss += loss.item() # compute total validation loss

                pred = output.argmax(dim=1).cpu().numpy() # Gets predicted class labels by taking the index of the highest logit
                y_true.extend(y_val.cpu().numpy()) # adds true labels to y_true
                y_pred.extend(pred) # adds predictions to y_pred

        # computes accuracy, f1 and average validation loss
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"[Fold {fold}]Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | Val Acc: {acc:.4f} | Val loss: {avg_val_loss:.4f} | F1: {f1:.4f}")
        if epoch == 0 or avg_val_loss < best_val_loss:
            # saves model with best f1 score
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


    # eveluating best model on validation data
    model.load_state_dict(best_model) # loading best model
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            output = model(X_batch)
            pred = output.argmax(dim=1).cpu().numpy()
            y_true.extend(y_batch.numpy())
            y_pred.extend(pred)
    
    # computing metrics for best model for current fold
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
    print(f"\nConfusion Matrix:\n{cm}")
    print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
    print(f"\nValidation Accuracy: {acc_final:.4f}")
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

# hyperparameters
def main():
    epochs = 20
    n_splits = 5
    lr = 0.00005
    patience = 3
    # train(epoch)

    # ==================== K-Fold Cross Validation ====================
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True) # Creates stratified k-fold splitter
    metrics_all_folds = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, Y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        result = train(X_train, y_train, X_val, y_val, fold, epochs, patience=patience, lr=lr)
        metrics_all_folds.append(result) # appending results to metrics_all_folds

    avg_f1 = np.mean([m['f1'] for m in metrics_all_folds])
    avg_acc = np.mean([m['accuracy'] for m in metrics_all_folds])

    print("\n==== Final Cross-Validation Report ====")

    print("\n-------------------------------------------------------")
    print(f"  Average F1 over {n_splits} folds:           {avg_f1:.4f}")
    print(f"  Average accuracy over {n_splits} folds:     {avg_acc:.4f}")
    print("-------------------------------------------------------")

if __name__ == '__main__':
    main()