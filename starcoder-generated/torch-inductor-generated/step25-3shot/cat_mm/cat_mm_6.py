
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mul(torch.t(x1), x2)
        v2 = torch.cat([v1, v1, v1, v1, v1, v1], 1)
        return v2
# Inputs to the model
x1 = torch.rand((2,4))
x2 = torch.rand((1,3))
