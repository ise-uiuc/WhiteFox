
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = torch.nn.functional.linear(x1, torch.randn(4, 2, 2), None)
        v2 = torch.cat([v1, v1], dim=2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
