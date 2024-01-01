
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
    def forward(self, x1):
        x2 = self.conv1(x1)
        y1 = self.conv2(x2)
        y1 = torch.add(torch.add(y1, y1), y1)
        y2 = self.conv2(x1)
        return torch.add(torch.add(y2, y2), y2)
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
