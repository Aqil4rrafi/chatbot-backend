import torch
import torch.nn as nn

class ChatNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ChatNet, self).__init__()
        # Input Layer ke Hidden Layer 1
        self.l1 = nn.Linear(input_size, hidden_size)
        # Hidden Layer 1 ke Hidden Layer 2
        self.l2 = nn.Linear(hidden_size, hidden_size)
        # Hidden Layer 2 ke Output Layer
        self.l3 = nn.Linear(hidden_size, num_classes)
        # Fungsi Aktivasi ReLU
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.l1(x))
        out = self.relu(self.l2(out))
        out = self.l3(out)
        return out