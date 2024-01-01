
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3, stride=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
