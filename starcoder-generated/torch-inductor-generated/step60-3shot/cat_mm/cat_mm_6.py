
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([a for a in x], 1)
# Inputs to the model
x1 = torch.randn(1, 5)
x2 = torch.randn(1, 5)
