
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        v1 = torch.mm(x, y)
        return torch.concat([v1, v1, v1, v1], 1)
# Inputs to the model
x1 = torch.randn(50,)
x2 = torch.randn(6, 1)
