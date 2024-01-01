
import torch

def foo(m):
    out1, out2 = m(tensor1, tensor2)
    return out1 + out2, out1 - out2

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = conv1 = torch.nn.Conv2d(3,2,3)
        self.fc1 = fc1 = torch.nn.Linear(16, 8)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        return out
# Inputs to the model
tensor1 = torch.randn(2, 3, 16, 16)
tensor2 = torch.randn(2, 3, 16, 16)
