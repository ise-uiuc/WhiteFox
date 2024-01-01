
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(3))
    def forward(self, x1):
        v1 = self.weight * x1
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
