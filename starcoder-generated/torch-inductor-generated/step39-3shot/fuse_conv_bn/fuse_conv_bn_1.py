
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv1.bias = torch.nn.Parameter(torch.randn(3))
        self.bn1 = torch.nn.BatchNorm2d(3)
        # self.conv1.register_buffer("running_mean", torch.zeros(3))
        self.conv2 = torch.nn.Conv2d(3, 3, 3, bias=False)
        self.conv2.bias = torch.nn.Parameter(torch.randn(3))
        self.bn2 =torch.nn.BatchNorm2d(3)
        # self.conv2.register_buffer("running_mean", torch.zeros(3))
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 6, 6)
