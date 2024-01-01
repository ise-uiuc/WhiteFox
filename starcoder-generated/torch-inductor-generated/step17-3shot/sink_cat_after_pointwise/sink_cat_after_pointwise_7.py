
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc0 = torch.nn.Linear(12, 16)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(16, 2)
    def forward(self, x):
        x = self.fc0(x)
        x = self.relu(x)
        x = self.fc1(x)
        return x
# Inputs to the model
x = torch.randn(4, 12)
