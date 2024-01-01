
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = torch.zeros(10, 10)
        return torch.cat([v + v * v])
# Inputs to the model
x1 = torch.randn(1, 10)
