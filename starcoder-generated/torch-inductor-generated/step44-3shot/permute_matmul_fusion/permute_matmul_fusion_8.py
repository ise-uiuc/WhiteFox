
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = x1.permute(0, 2, 1)
        v3 = torch.bmm(x1, v1)
        v4 = torch.bmm(x2, v1)
        v5 = torch.bmm(x2, v3)
        v6 = torch.bmm(x1, v4)
        v7 = torch.randn(1, 3, 3)
        return torch.tanh(v7)
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
