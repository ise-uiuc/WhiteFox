
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, (1, 1), stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(6, 16, (3, 3), stride=2, padding=(1, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(torch.cat((x1, v1), dim=1))
        v3 = self.conv1(v2)
        v4 = self.conv2(torch.cat((x1, v3), dim=1))
        v5 = self.conv1(v4)
        v6 = torch.sigmoid(v5)
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
