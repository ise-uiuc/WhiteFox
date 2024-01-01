
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 3, 3, stride=2, padding=1, bias=False)
        self.conv2d = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(3, 3, 3, padding=1, bias=False)
    def forward(self, x):
        x = self.conv2(x)
        y = self.conv3(self.conv2d(x))
        return y
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
