
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(96, 2, 5, stride=1, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 96, 224, 224)
