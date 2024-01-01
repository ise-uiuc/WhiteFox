
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, (1, 3), stride=1, padding=(0, 1))
        self.conv2 = torch.nn.Conv2d(32, 4, (3, 1), stride=1, padding=(1, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 3, 5)
