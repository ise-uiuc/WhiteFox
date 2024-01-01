
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(5, 5, 3, stride=3, padding=3)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = -0.5 - v1
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 5, 96, 96)
