
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v4 = x1.permute(0, 2, 1)
        v5 = torch.bmm(x2, v4)[..., 0]
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
