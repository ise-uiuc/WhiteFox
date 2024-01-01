
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 32, 3, stride=2, padding=1)
    def forward(self, x2):
        v1 = self.conv1(x2)
        v2 = v1 - torch.scalar_tensor(1.0)
        v3 = v2 - torch.randn(8)
        return v3
# Inputs to the model
x2 = torch.randn(8, 32, 6, 6)
