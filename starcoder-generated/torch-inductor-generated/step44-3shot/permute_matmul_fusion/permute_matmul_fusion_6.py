
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x0, x1):
        v0 = x1.permute(0, 2, 1)
        v0 = x0.permute(0, 2, 1)
        v0 = x1.permute(0, 2, 1)
        v1 = x0.permute(0, 2, 1)
        v2 = torch.bmm(v0, v0)
        v3 = torch.bmm(v1, v1)
        return torch.tanh(v2)
# Inputs to the model
x0 = torch.randn(1, 2, 2)
x1 = torch.randn(1, 2, 2)
