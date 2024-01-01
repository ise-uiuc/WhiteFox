
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * -1
        v3 = self.conv1(v2)
        v4 = v2 + v3
        v5 = F.relu(v4)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
