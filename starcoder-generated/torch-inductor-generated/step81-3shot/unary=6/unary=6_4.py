
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.numel()
        v2 = torch.sum(x1)
        v3 = torch.square(x1)
        v4 = torch.norm(x1)
        v5 = torch.sum(v3 + v4)
        v6 = torch.sqrt(v5)
        v7 = torch.abs(v6)
        v8 = v7 / v2
        return v8
# Inputs to the model
x1 = torch.randn(1, 47616)
