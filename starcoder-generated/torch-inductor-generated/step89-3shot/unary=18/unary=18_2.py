
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (3, 3), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 16, (3, 3), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.tanh(v2)
        v4 = torch.relu(v3)
        v5 = self.conv2(v4)
        v6 = torch.sigmoid(v5)
        v7 = torch.sigmoid(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
