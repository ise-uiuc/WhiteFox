
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 1, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = torch.relu(x1)
        v2 = torch.relu(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 8, 1, 1)
