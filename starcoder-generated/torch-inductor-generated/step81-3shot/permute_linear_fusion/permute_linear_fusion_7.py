
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
    def forward(self, x1):
        x2 = torch.where((x1 > 0), torch.full_like(x1, 0.99), x1)
        x3 = torch.where((x1 > 1), torch.full_like(x1, 0.01), x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 2)
