
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(4, 4, 1, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.relu(v1)
        v3 = self.conv2(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x = torch.randn(2, 16, 16, 16)
