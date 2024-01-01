
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x2)
        v2 = abs(v1)
        v3 = self.conv1(v2)
        v4 = torch.relu(v3)
        return torch.conv2d(v4, torch.cat([torch.zeros(3, 1, 1), torch.rand(3, 1, 3), torch.zeros(2, 2, 3)], 1))
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
x2 = torch.randn(1, 3, 32, 32)
