
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x2
        v3 = x1
        v4 = torch.bmm(v1, v2)
        x11 = v1 * v2
        x12 = v3 * v2
        x13 = v2 * v2
        return (x11, x12, x13, v4)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
