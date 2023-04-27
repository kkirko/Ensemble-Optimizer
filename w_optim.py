import torch
import torch.nn as nn

torch.manual_seed(42)

class WeightNet(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.5):
        super(WeightNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, output_size)

        nn.init.constant_(self.fc5.weight, 1/3)
        nn.init.constant_(self.fc5.bias, 0)

        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)
    
        x = self.fc4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)
        
        x = self.fc5(x)
        x = torch.softmax(x, dim=1)
        return x
