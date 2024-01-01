
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = x1.permute(0, 2, 1)
        v4 = torch.bmm(v1, v2)
        v5 = torch.bmm(v3, v4)
        v6 = torch.bmm(v3, v5)
        return v6
# Inputs to the model
x1 = torch.randn(4, 1, 7)
x2 = torch.randn(4, 10, 7)
