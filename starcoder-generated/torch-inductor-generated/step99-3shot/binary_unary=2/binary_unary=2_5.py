
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 256, bias=True)
        self.fc2 = torch.nn.Linear(256, 512, bias=True)
        self.fc3 = torch.nn.Linear(512, 512, bias=True)
        self.fc4 = torch.nn.Linear(512, 256, bias=True)
    def forward(self, x1):
        v1 = F.relu(self.fc1(x1))
        v2 = F.relu(self.fc2(v1))
        v3 = F.relu(self.fc3(v2))
        v4 = F.relu(self.fc4(v3))
        v5 = v4 - 1.08
        x2 = F.relu(v5)
        return x2
# Inputs to the model
x1 = torch.randn(1, 3)
