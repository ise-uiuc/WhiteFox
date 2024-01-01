
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
	# Please use torch.nn.Conv2d() to apply pointwise convolution with kernel size 1 and padding 1
        self.covn = torch.nn.Conv2d(3, 128, 1, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(128, 128, 3,padding=1, dilation=1)
        self.flatten = torch.nn.Flatten()
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(1152, 144)
        self.conv3 = torch.nn.Conv2d(144, 144, 3, padding=1, dilation=1, stride=1, groups=1)
    def forward(self, x):
        t2 = self.tanh(self.conv3(self.linear(self.tanh(self.flatten(self.conv2(self.tanh(self.covn(x))))))))
        return t2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
