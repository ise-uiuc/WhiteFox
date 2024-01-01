
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6(inplace=True) # ReLU6 activation
        self.conv_1 = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.conv_2 = torch.nn.Conv2d(8, 7, 1, stride=1, padding=1)
        self.conv_3 = torch.nn.Conv2d(7, 5, 1, stride=1, padding=1)
        self.conv_4 = torch.nn.Conv2d(5, 1, 1, stride=1, padding=1)
    def forward(self, x1):
        # Construct the 2nd output tensor of the pattern using a pointwise convolution with kernel size 1 (equivalent to a ReLU6 activation)
        x1 = self.relu6(x1)
        # Construct the 1st output tensor of the pattern using a pointwise convolution with kernel size 1
        x1 = self.conv_1(x1) # Conv_1
        # Construct the 2nd output tensor of the pattern using a pointwise convolution with kernel size 1
        x1 = self.conv_2(x1) # Conv_2
        # Construct the 3rd output tensor of the pattern using a pointwise convolution with kernel size 1
        x1 = self.conv_3(x1) # Conv_3
        # Construct the 4th output tensor of the pattern using a pointwise convolution with kernel size 1
        x1 = self.conv_4(x1) # Conv_4
        return x1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
