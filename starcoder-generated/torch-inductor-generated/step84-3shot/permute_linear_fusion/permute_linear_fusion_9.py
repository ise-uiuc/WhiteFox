
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
    def forward(self, x1):
        return torch.tanh(x1)
# Inputs to the model
x1 = torch.randn(1, 1)
