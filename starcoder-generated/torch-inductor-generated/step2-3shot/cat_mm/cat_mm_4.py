
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        return torch.cat([x1, x1, x1], 1)
# Inputs to the model
x1 = torch.randn(4, 2)
x2 = torch.randn(4, 1)
