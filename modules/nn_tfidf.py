import torch
import torch.nn as nn
import torch.nn.functional as F


class NNTfidf(nn.Module):

    def __init__(self):
        super(NNTfidf, self).__init__()
        self.fc1 = nn.Linear(300, 100)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(100, 50)
        self.dropout2 = nn.Dropout(0.25)
        self.fc3 = nn.Linear(50, 10)
        self.dropout3 = nn.Dropout(0.25)
        self.fc4 = nn.Linear(10, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = F.softmax(x)
        return x