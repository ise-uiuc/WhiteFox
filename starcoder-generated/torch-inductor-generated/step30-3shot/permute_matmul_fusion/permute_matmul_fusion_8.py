
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.bmm(x2.permute(0, 2, 1), x1)
        v2 = torch.bmm(x1, v1)
        v3 = torch.bmm(v1, v2)
        v4 = torch.bmm(v2, v3)
        v5 = torch.bmm(v3, v4)
        return (v1, v2, v3, v4, v5)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
