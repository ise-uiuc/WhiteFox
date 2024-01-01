
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.zeros(3, 2, 2), torch.zeros(3, 1))
        v2 = v1.permute(0, 2, 1)
        v3 = x1 - v2
        v4 = x1 / v1
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
