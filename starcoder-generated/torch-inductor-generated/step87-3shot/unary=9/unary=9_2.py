
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(5, 6, 1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        x1 = torch.add(v1, 3)
        x2 = x1.clamp(0, 6)
        x3 = x2.div(6)
        x4 = self.conv2(x3)
        x5 = x4 + 3
        x6 = x5.clamp(0, 6)
        x7 = x6.div(6)
        return x7 
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
