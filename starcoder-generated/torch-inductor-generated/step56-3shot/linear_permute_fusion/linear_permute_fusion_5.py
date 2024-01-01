
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v4 = x1
        v8 = torch.sin(x1)
        v1 = torch.sin(v4) + v8
        v2 = v1.permute(0, 2, 1)
        v3 = torch.reciprocal(x1)
        return v2 * v3
# Inputs to the model
x1 = torch.randn(1, 2, 2, device='cpu')
