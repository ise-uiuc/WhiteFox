
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 3, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 4, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 12
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = v4 - 13
        v6 = torch.squeeze(v5, 0)
        return v6
# Inputs to the model
x1 = torch.randn(1, 4, 56, 56)
