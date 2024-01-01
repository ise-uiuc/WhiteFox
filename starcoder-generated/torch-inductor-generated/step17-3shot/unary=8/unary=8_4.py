
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4, 4, 1, stride=1)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(4, 4, 3, stride=2, padding=0, dilation=1)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x1):
        y1 = self.conv1(x1)
        y1 = self.relu1(y1)
        y2 = self.conv2(y1)
        y2 = self.relu2(y2)
        return y2
# Inputs to the model
x1 = torch.randn(1, 4, 128, 128)
