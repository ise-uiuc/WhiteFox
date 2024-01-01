
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 128, 3, stride=1, padding=1)
        self.c2 = torch.nn.Conv2d(128, 128, 1)
        self.c3 = torch.nn.Conv2d(128, 32, 1)
    def forward(self, x1, x2):
        y1 = self.c1(x1)
        y2 = self.c2(y1)
        y3 = self.c3(y2)
        y4 = y3 + x2
        y5 = F.relu(y4)
        return y5
# Inputs to the model
x1 = torch.randn(2, 1, 28, 28)
x2 = torch.randn(2, 32, 28, 28)
