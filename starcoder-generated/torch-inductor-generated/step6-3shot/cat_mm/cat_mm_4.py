
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        return torch.cat([torch.tensor([[1., 2., 3., 4.]]), v2, v2, v2, v2, v2], 1)
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(4, 4)
