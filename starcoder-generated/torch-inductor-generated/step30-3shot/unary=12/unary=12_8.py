
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = x1.sigmoid()
        v2 = v1 * v1
        return v2
# Inputs to the model
x1 = torch.randn(16, 2, 32, 32)
