
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 32, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv2(torch.sigmoid(v1) + 2 * x2)
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32).to('cpu')
x2 = torch.randn(1, 256, 2, 2).to('cpu')
