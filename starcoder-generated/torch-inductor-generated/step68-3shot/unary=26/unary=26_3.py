
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 12, stride=5, padding=3, bias=True)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(8, 1, 20, stride=1, padding=2, bias=True)
    def forward(self, x1):
        y0 = torch.nn.functional.interpolate(x1, scale_factor=[0.5, 0.5])
        y1 = self.conv1(y0)
        y2 = y1 + 0.1689
        y3 = self.relu(y2)
        y4 = torch.nn.functional.interpolate(y3, scale_factor=[0.1, 0.1])
        y5 = self.conv2(y4)
        y6 = y5 * 0.2548
        y7 = torch.nn.functional.interpolate(y6, scale_factor=[2.0, 2.0])
        return y7
# Inputs to the model
x1 = torch.randn(1, 1, 30, 50)
