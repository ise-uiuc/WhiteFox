
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=1)
        self.conv2 = torch.nn.Conv2d(1, 1, 3, stride=1)
        # self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=1, dilation=3) # This is the original layer which does not have bias
        self.conv3 = torch.nn.Conv2d(1, 1, 3, stride=1, padding=3, dilation=3, bias=False) # This one has bias
    def forward(self, x):
        t = torch.tanh(self.conv1(x))
        y = self.conv2(t)
        z = self.conv3(t)
        return torch.tanh(y)
# Inputs to the model
x = torch.randn(1, 1, 64, 62)
