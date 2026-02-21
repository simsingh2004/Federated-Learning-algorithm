import torch
from torch import optim
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import torchmetrics
from flwr_datasets import FederatedDataset           
from flwr_datasets.partitioner import IidPartitioner    
from torchvision.transforms import Compose, ToTensor, Normalize

#SOME REASON HUGGINGFACE DATASET IS GIVING ISSUE SO CREATED A VIRTUAL ENVIORNMENT WITH PYTHON 3.12 AND INSTALLED ALL THE LIBRARIES AND IT WORKED FINE!!!

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fds = None
def load_data(partition_id: int, num_partitions: int, batch_size: int = 60):
   

    global fds  
    if fds is None:  
        partitioner = IidPartitioner(num_partitions=num_partitions)  
        fds = FederatedDataset(  
            dataset="ylecun/mnist",  
            partitioners={"train": partitioner}, 
        )

    partition = fds.load_partition(partition_id)  
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)  
    pytorch_transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])  

    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]  
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms) 

    
    train_loader = DataLoader(partition_train_test["train"], batch_size=batch_size, shuffle=True)  
    test_loader = DataLoader(partition_train_test["test"], batch_size=batch_size, shuffle=False)   

    return train_loader, test_loader


class CNN(nn.Module):
   def __init__(self, in_channels, num_classes):

       super(CNN, self).__init__()
       self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
       self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
       self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
       self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

   def forward(self, x):
       x = F.relu(self.conv1(x))  
       x = self.pool(x)           
       x = F.relu(self.conv2(x))  
       x = self.pool(x)         
       x = x.reshape(x.shape[0], -1)  
       x = self.fc1(x)           
       return x
   


def train(model, train_loader, device, num_epochs=10, lr=1e-3):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"): 
            images = batch["image"].to(device) 
            labels = batch["label"].to(device)  

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    return running_loss / len(train_loader)


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device) 
            labels = batch["label"].to(device) 

            outputs = model(images)

            
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            num_batches += 1

            
            preds = torch.argmax(outputs, dim=1)
            accuracy.update(preds, labels)

    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, accuracy.compute().item()


def load_model():
    return CNN(in_channels=1, num_classes=10).to(device)

    
if __name__ == "__main__":
    cnn = load_model()

    
    train_loader, test_loader = load_data(partition_id=0, num_partitions=1)  

    loss = train(cnn, train_loader, device, num_epochs=5)
    test_loss, acc = test(cnn, test_loader, device)

    print(f"Train loss: {loss:.5f}, Test loss: {test_loss:.5f}, Accuracy: {acc:.3f}")

