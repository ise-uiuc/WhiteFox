
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 32)
        self.tanh = torch.nn.Tanh()
    def forward(self, x1):
        v1 = self.tanh(self.linear(x1))
        return v1

# Inputs to the model
x1 = torch.randn(1, 8)
