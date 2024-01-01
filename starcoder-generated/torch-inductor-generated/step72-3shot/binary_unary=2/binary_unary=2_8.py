
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = v2 - 0.5
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 2048, 2048)
