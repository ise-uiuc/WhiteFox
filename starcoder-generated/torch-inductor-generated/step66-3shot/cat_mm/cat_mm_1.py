
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y = []
        y.append(x1)
        return torch.cat(y, 4)
# Inputs to the model
x1 = torch.randn(1, 2, 2, 1)
