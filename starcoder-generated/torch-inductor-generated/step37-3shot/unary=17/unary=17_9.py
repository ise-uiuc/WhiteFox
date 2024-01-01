
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 512, 16, padding=0, stride=1)
        self.conv2 = torch.nn.Conv2d(512, 2, 3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 128, 1, 1)
