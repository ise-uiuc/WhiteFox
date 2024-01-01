
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([x1, x1, x1], 1)
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 3)
