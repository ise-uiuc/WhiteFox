
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(2, 1)
        self.relu2 = nn.ReLU6()
    def forward(self, x2):
        x1 = self.fc1(x2)
        x1 = self.relu1(x1)
        x1 = self.fc2(x1)
        x1 = self.relu2(x1)
        return x1
# Inputs to the model
x2 = torch.randn(20, 2)
