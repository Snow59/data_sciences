
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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

# Modèle
class InsolesMocapClassifier(nn.Module):
    def __init__(self, insole_input_size=50, mocap_input_size=129, hidden_size=384, num_classes=13):
        super().__init__()
        self.lstm_insole = nn.LSTM(insole_input_size, hidden_size, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.lstm_mocap = nn.LSTM(mocap_input_size, hidden_size, num_layers=2, dropout=0.3, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(4 * hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x_insole, x_mocap):
        out_insole, _ = self.lstm_insole(x_insole)
        out_mocap, _ = self.lstm_mocap(x_mocap)
        out_insole = out_insole[:, -1, :]
        out_mocap = out_mocap[:, -1, :]
        combined = torch.cat((out_insole, out_mocap), dim=1)
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

    # Class weights
    all_labels = np.array(dataset.labels[:]).squeeze().astype(int)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_labels), y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    num_epochs = 50
    best_val_loss = float('inf')
    trigger = 0
    patience = 10

    train_loss_log, val_loss_log, val_acc_log = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_insole, x_mocap, labels in train_loader:
            x_insole, x_mocap, labels = x_insole.to(device), x_mocap.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(x_insole, x_mocap)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_insole, x_mocap, labels in val_loader:
                x_insole, x_mocap, labels = x_insole.to(device), x_mocap.to(device), labels.to(device)
                outputs = model(x_insole, x_mocap)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        scheduler.step(avg_val_loss)

        train_loss_log.append(avg_train_loss)
        val_loss_log.append(avg_val_loss)
        val_acc_log.append(val_acc)

        print(f"Epoch {epoch+1:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trigger = 0
            torch.save(model.state_dict(), 'model_best_advanced.pth')
        else:
            trigger += 1
            if trigger >= patience:
                print("⏹ Early stopping triggered.")
                break

    # Courbes
    plt.plot(train_loss_log, label="Train Loss")
    plt.plot(val_loss_log, label="Val Loss")
    plt.plot(val_acc_log, label="Val Accuracy")
    plt.legend()
    plt.grid()
    plt.title("Training Progress (Advanced)")
    plt.show()

if __name__ == "__main__":
    train_model()
