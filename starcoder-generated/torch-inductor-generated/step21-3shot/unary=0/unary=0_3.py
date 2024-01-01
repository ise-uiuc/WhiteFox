
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 2, 1, stride=1, padding=0)
        self.conv_2 = torch.nn.Conv2d(6, 2, 1, stride=1, padding=0)
        self.softmax = torch.nn.Softmax(1)
    def forward(self, x):
        v1 = self.softmax(self.conv(x))
        v2 = self.softmax(self.conv_2(x))
        v3 = v1 * v2
        return v3
# Inputs to the model
x = torch.randn(1, 6, 3, 64)
