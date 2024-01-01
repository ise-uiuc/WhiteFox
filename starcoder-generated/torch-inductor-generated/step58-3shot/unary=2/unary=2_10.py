
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1 * 0.55
        v2 = torch.ones_like(x1, requires_grad=True)
        v3 = torch.ones(1)
        v4 = v3 * x1 * v2 * v2 * v2
        v5 = torch.ones_like(x1)
        v6 = v4 * torch.dot(v1, v5)/v2
        return v6
# Inputs to the model
x1 = torch.randn(10, 10)
