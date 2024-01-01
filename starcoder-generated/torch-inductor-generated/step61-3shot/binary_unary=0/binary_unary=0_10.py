
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.randn(1, 16, 64, 64)
        v7 = v1 + v2
        return v7
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
