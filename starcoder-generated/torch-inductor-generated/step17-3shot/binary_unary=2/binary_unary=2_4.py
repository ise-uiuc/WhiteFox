
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 12
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
