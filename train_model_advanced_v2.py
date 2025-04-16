
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

# Dataset
class InsolesMocapDataset(Dataset):
    def __init__(self, insole_path, mocap_path, label_path, augment=False):
        self.insoles = h5py.File(insole_path, 'r')['my_data']
        self.mocap = h5py.File(mocap_path, 'r')['my_data']
        self.labels = h5py.File(label_path, 'r')['my_data']
        self.augment = augment

    def __len__(self):
        return len(self.insoles)

    def __getitem__(self, idx):
        insole_seq = torch.tensor(self.insoles[idx].reshape(100, 50), dtype=torch.float32)
        mocap_seq = torch.tensor(self.mocap[idx].reshape(100, 129), dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()

        insole_seq = torch.nan_to_num(insole_seq, nan=0.0, posinf=0.0, neginf=0.0)
        mocap_seq = torch.nan_to_num(mocap_seq, nan=0.0, posinf=0.0, neginf=0.0)
        insole_seq = torch.clamp(insole_seq, -10, 10)
        mocap_seq = torch.clamp(mocap_seq, -10, 10)
        insole_seq = (insole_seq - insole_seq.mean()) / (insole_seq.std() + 1e-8)
        mocap_seq = (mocap_seq - mocap_seq.mean()) / (mocap_seq.std() + 1e-8)

        if self.augment:
            insole_seq += torch.randn_like(insole_seq) * 0.01
            mocap_seq += torch.randn_like(mocap_seq) * 0.01

        return insole_seq, mocap_seq, label

# Attention module
class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        weighted = torch.sum(x * weights, dim=1)
        return weighted

# Modèle
class InsolesMocapClassifier(nn.Module):
    def __init__(self, insole_input_size=50, mocap_input_size=129, hidden_size=384, num_classes=13):
        super().__init__()

        self.cnn_insole = nn.Conv1d(insole_input_size, insole_input_size, kernel_size=3, padding=1)
        self.cnn_mocap = nn.Conv1d(mocap_input_size, mocap_input_size, kernel_size=3, padding=1)

        self.lstm_insole = nn.LSTM(insole_input_size, hidden_size, num_layers=2, dropout=0.3,
                                   batch_first=True, bidirectional=True)
        self.lstm_mocap = nn.LSTM(mocap_input_size, hidden_size, num_layers=2, dropout=0.3,
                                  batch_first=True, bidirectional=True)

        self.norm = nn.LayerNorm(2 * hidden_size)
        self.attn = SelfAttention(2 * hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(2 * hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_insole, x_mocap):
        x_insole = self.cnn_insole(x_insole.permute(0, 2, 1)).permute(0, 2, 1)
        x_mocap = self.cnn_mocap(x_mocap.permute(0, 2, 1)).permute(0, 2, 1)

        out_insole, _ = self.lstm_insole(x_insole)
        out_mocap, _ = self.lstm_mocap(x_mocap)

        out_insole = self.norm(out_insole)
        out_mocap = self.norm(out_mocap)

        out_insole = self.attn(out_insole)
        out_mocap = self.attn(out_mocap)

        combined = (out_insole + out_mocap) / 2
        return self.fc(combined)

# Entraînement
def train_model():
    insole_path = 'train_insoles.h5'
    mocap_path = 'train_mocap.h5'
    label_path = 'train_labels.h5'

    dataset = InsolesMocapDataset(insole_path, mocap_path, label_path, augment=True)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    val_dataset.dataset.augment = False

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InsolesMocapClassifier().to(device)

    # Poids des classes
    all_labels = np.array(dataset.labels[:]).squeeze().astype(int)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    patience = 10
    trigger = 0

    for epoch in range(1, 30):
        model.train()
        total_loss = 0
        for xi, xm, y in train_loader:
            xi, xm, y = xi.to(device), xm.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(xi, xm)
            loss = criterion(output, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for xi, xm, y in val_loader:
                xi, xm, y = xi.to(device), xm.to(device), y.to(device)
                output = model(xi, xm)
                val_loss += criterion(output, y).item()
                _, pred = torch.max(output, 1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger = 0
            torch.save(model.state_dict(), 'model_best_advanced_v2.pth')
        else:
            trigger += 1
            if trigger >= patience:
                print("⏹ Early stopping triggered.")
                break

if __name__ == "__main__":
    train_model()
