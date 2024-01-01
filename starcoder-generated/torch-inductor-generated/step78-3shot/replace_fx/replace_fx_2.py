
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        x2 = torch.rand_like(x1)
        x3 = x2.flatten(1)
        return torch.cat((x3, x2), 0).view((-1, 1, 4, 4))
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
x2 = torch.randn(1, 5, 4, 4)
