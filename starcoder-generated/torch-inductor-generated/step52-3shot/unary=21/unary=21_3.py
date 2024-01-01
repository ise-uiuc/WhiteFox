
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 3, kernel_size=(192, 29, 4), stride=(192, 29, 4), groups=1)
        self.conv2 = torch.nn.Conv2d(3, 3, kernel_size=(325, 6, 34), stride=(325, 6, 34), groups=1, dilation=253)
    def forward(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        x4 = torch.tanh(x3)
        return x4
# Inputs to the model
x1 = torch.randn(1, 1, 352, 946)
