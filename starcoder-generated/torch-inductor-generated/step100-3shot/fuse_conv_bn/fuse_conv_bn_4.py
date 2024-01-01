
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(4, 5, 1)
        self.conv2 = torch.nn.Conv2d(5, 8, 1)
        self.conv3 = torch.nn.Conv2d(8, 1, 1)
    def forward(self, x):
        y1 = self.relu1(x)
        y2 = self.conv1(y1)
        y3 = self.conv2(y2)
        y4 = self.conv3(y3)
        return y4
# Inputs to the model
x = torch.randn(1, 4, 10, 10)
