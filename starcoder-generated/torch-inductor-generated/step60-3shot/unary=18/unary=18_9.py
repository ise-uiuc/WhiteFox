
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 32, 1, stride=1, padding=0, bias=False)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 128, 80, 48)
