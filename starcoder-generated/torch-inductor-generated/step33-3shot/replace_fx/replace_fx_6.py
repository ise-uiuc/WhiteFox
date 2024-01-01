
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parameter0 = torch.nn.Parameter(torch.tensor((-0.0535, -1.3267, -1.1838, 1.0735), dtype=torch.float32))
    def forward(self, x1):
        x2 = torch.nn.functional.dropout(x1)
        x3 = x1 + 1
        x4 = self.parameter0 - x3
        return x4
# Inputs to the model
x1 = torch.randn(1, 2, 2)
