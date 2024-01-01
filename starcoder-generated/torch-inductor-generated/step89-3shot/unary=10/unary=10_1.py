
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Number of input channels in the image (3)
        # Number of output channels produced by the convolution (16)
        # Size of the convolution kernel (55)
        self.conv0 = torch.nn.Conv2d(3, 16, 5, stride=1, padding=0)
        # Number of input channels (16)
        # Number of output channels (32)
        # Size of the convolution kernel (33)
        self.conv1 = torch.nn.Conv2d(16, 32, 3, stride=1, padding=0)
        # Number of input channels (32)
        # Number of output channels (32)
        # Size of the convolution kernel (33)
        # Whether the convolution should be used in a transposed (True) or forward (False) pass
        self.conv2 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
        # Number of input channels (32)
        # Number of output channels (32)
        # Size of the convolution kernel (33)
        # Whether the convolution should be used in a transposed (True) or forward (False) pass
        self.conv3 = torch.nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=1)
 
    def forward(self, x1):
        v0 = self.conv0(x1)
        v1 = self.conv1(v0)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
