
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(308, 200, (7, 7), stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(200, 1, (7, 7), stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 308, 7, 7)
