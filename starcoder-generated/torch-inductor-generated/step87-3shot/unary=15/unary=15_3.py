
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 256, (1, 11), stride=1, padding=(0, 5))
        self.conv2 = torch.nn.Conv2d(4, 256, (1, 11), stride=1, padding=(0, 5))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.relu(v1)
        v3 = self.conv2(x1)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(2, 4, 256, 256)
