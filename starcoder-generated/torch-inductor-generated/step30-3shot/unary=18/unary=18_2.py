
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, (64, 32), stride=(62, 30), padding=(0, 0))
        self.conv2 = torch.nn.Conv2d(64, 256, (16, 21), stride=(16, 21), padding=(0, 0))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = torch.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 512, 211)
