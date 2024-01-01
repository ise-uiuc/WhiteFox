
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(120, 84)
    def forward(self, x):
        v1 = self.fc1(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(64, 16, 50)
