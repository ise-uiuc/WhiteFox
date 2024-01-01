
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x2.permute(0, 2, 1)
        v2 = torch.tensor(42)
        v2 = v2.permute(0, 2, 1)
        v3 = torch.tensor(43)
        v3 = v3.permute(0, 1, 2)
        v4 = torch.bmm(x1, v1)
        v4 = v4.permute(1, 2, 0)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2, 2)
x2 = torch.randn(2, 2, 2)
