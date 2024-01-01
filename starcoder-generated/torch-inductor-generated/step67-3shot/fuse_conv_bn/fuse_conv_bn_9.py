
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        class ConvBNReLUPool2DFactory():
            def __init__(self):
                self.conv = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.bn = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=False)
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            def forward(self, x):
                x = self.conv(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.pool(x)
                return x

        self.block1 = ConvBNReLUPool2DFactory()
        self.block2 = ConvBNReLUPool2DFactory()

        def identity_function(x):
            return x
        self.identity_block = identity_function
    def forward(self, input):
        return self.block2(self.block1(input)) + self.identity_block(input)
# Inputs to the model
x = torch.randn(1, 64, 32, 32)
