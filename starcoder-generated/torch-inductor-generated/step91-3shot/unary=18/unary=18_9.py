
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.float()
        v2 = v1.type(torch.float64)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
