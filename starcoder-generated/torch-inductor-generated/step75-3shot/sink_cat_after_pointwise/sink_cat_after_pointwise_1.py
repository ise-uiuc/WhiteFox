
class Model(torch.nn.Module):
    def __init__(self, m, n):
        super().__init__()
        self.fc = torch.nn.Linear(n, m).to(torch.float32)
    def forward(self, x):
        y = self.fc(x)
        return y
# Inputs to the model
x = torch.randn(1, 10, dtype=torch.float64)
