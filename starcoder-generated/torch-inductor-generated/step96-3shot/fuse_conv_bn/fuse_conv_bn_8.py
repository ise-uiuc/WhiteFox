
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(3, 4, 3)
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        # Pass the input through the convolution layer
        x = self.conv2d(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(5, 3, 10, 10)
