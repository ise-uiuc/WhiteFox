
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x2, x1):
        v2 = x2.permute(0, 2, 1)
        v1 = x1.permute(0, 2, 1)
        s = torch.add(v2, v1)[0][0][0]
        res = v2 * s
        return res
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
