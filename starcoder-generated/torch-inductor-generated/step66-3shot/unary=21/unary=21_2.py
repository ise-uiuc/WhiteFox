
class ModelHswish(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2a = torch.nn.Conv2d(3, 4, 3, stride=2, groups=2)
        self.conv2b = torch.nn.Conv2d(4, 4, 3, stride=1, groups=2)
        self.pool = torch.nn.AvgPool2d(3)
    def forward(self, x_in):
        x1 = self.conv2a(x_in)
        x2 = torch.nn.functional.relu6(x1)
        x3 = self.conv2b(x2)
        x4 = torch.nn.functional.relu6(x3)
        x5 = self.pool(x4)
        x6 = torch.mul(0.5, x5)
        return x6   
# Inputs to the model
x_in = torch.randn(2, 3, 64, 64)
