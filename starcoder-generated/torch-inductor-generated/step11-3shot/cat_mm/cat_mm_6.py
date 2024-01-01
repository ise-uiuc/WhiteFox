
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat(x, 1)
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
