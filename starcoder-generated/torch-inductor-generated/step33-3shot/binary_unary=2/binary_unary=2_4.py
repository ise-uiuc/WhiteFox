
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 9, stride=2, padding=4)
        self.conv2 = torch.nn.Conv2d(4, 8, 9, stride=4, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = F.relu(v1)
        v3 = v1 - -1
        v4 = self.conv2(v2.transpose(-1, -2)).transpose(-1, -2)
        v5 = -torch.tanh(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
