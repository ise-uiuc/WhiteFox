
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v1 = v1.reshape([100000, 100000])
        v2 = v1*3 + v1
        return torch.cat([v2, v1, v1 + v2], 1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 2)
