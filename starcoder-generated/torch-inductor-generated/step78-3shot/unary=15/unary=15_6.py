
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 1, (32, 32), stride=(1, 1), padding=(16, 16), dilation=(1, 1), groups=1)
    def forward(self, x1):
        v0 = x1
        v1 = self.conv(v0)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
