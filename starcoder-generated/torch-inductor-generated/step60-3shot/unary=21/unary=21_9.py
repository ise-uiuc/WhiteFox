
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 128, 256, 256, 16)
    def forward(self, x):
        y = self.linear(x)
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(64, 100)
