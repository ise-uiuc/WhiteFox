
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, (3, 3), stride=1, padding=(1, 1))
        self.conv2 = torch.nn.Conv2d(16, 15, (3, 3), stride=1, padding=(1, 1))
        self.conv3 = torch.nn.Conv2d(15, 16, (1, 1), stride=1, padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = self.conv2(v1)
        v5 = torch.relu(v1)
        v6 = self.conv3(v5)
        v7 = torch.relu(v6)
        return v7
# Inputs to the model
x1 = torch.randn(1, 1, 112, 112)
