
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(2, 0, 1)
        v2 = x1.permute(0, 2, 1)
        v3 = x1.permute(0, 1, 2)
        v4 = x1.permute(1, 0, 2)
        v5 = torch.bmm(v5, x1)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
