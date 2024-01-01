
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.reshape(x1, (3, 2, 1))
        v2 = torch.reshape(x2, (3, 2, 1))
        v3 = torch.bmm(v1, v2)
        v4 = torch.bmm(v3, v2)
        v5 = torch.bmm(v3, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 6)
x2 = torch.randn(1, 6)
x3 = torch.randn(1, 6)
