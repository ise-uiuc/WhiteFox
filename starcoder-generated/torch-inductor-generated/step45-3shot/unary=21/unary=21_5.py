
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 43, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(1, 86, 1, stride=1, padding=1)
    def forward(self, x1, x2):
        y1 = self.conv1(x1)
        y2 = self.conv2(x2)
        y3 = torch.tanh(y1 + y2)
        return y3
# Input to the model
x1 = torch.randn(1, 1, 20, 20)
# Input to the model
x2 = torch.randn(1, 1, 20, 20)
