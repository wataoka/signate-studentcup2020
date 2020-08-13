import torch
import torch.nn as nn
import torch.nn.functional as F


class NNTfidf(nn.Module):

    def __init__(self):
        super(NNTfidf, self).__init__()
        self.fc1 = nn.Linear(300, 50)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(50, 30)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(30, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.softmax(x)
        return x