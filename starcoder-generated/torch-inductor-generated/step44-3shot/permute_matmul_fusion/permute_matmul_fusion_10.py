
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x2):
        v1 = x0.permute(0, 2, 1)
        v2 = x2.permute(0, 2, 1)
        v3 = torch.bmm(v1, v2)
        v4 = torch.bmm(v2, v1)
        v5 = torch.randn(3, 3, 3)
        v6 = v3.permute(1, 0, 2)
        return torch.tanh(v5)
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
