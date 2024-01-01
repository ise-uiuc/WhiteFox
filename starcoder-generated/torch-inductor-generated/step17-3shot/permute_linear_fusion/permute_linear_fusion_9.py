
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        v1 = x1.permute(0, 2, 1)
        return torch.mul(self.tanh(v1), self.linear(v1))
# Inputs to the model
x1 = torch.randn(1, 2, 2)
