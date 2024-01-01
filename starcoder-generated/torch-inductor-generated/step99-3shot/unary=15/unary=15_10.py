
import torch.nn.functional as F
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(1, 64)
        self.fc2 = torch.nn.Linear(64, 10)
    def forward(self, x1):
        v1 = self.fc1(x1)
        v2 = F.relu(v1)
        v3 = self.fc2(v2)
        v4 = torch.log_softmax(v3)
        return v4
# Inputs to the model
x1 = torch.randn(5, 1)
