
import torch
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
        self.dropout1 = torch.nn.Dropout2d(p=0.25)
        self.avgpool1 = torch.nn.AvgPool2d(2, stride=2)
        self.dropout2 = torch.nn.Dropout2d(p=0.5)
        self.fc1 = torch.nn.Linear(4480, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.dropout3 = torch.nn.Dropout2d(p=0.25)
        self.fc3 = torch.nn.Linear(84, 10)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.dropout1(x2)
        x4 = self.avgpool1(x3)
        x5 = self.dropout2(x4)
        x6 = torch.flatten(x5, 1)
        x7 = self.fc1(x6)
        x8 = self.fc2(x7)
        x9 = self.dropout3(x7)
        x10 = self.fc3(x9)
        return x2, x9, x10
# Inputs to the model
x1 = torch.randn(1, 100, 100)
