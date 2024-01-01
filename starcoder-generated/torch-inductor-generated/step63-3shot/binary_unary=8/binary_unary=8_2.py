
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 4, 3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 4, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        w1 = self.conv2(v1)
        v2 = self.conv1(x1)
        w2 = self.conv2(v2)
        v3 = torch.relu(w1)
        w3 = torch.relu(w2)
        x2 = v3 + w3
        v4 = self.conv1(x2)
        w4 = self.conv2(v4)
        v5 = torch.relu(w4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 32, 32)
