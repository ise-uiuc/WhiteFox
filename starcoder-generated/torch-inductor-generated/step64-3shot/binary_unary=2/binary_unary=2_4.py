
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 2, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.0
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 32, 32)
