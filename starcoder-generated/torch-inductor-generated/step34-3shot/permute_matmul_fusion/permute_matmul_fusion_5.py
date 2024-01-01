
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v0 = x2.permute(0, 2, 1)
        x = torch.bmm(x1, v0)
        v1 = x.permute(2, 0, 1)
        v2 = v0.permute(2, 1, 0)
        x = torch.bmm(v2, v1)
        return (v0, v1, v2, x)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
