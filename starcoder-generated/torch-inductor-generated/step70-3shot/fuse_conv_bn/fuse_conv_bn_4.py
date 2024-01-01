
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        torch.manual_seed(1)
        self.norm1 = torch.nn.BatchNorm2d(1, affine=False)
        torch.manual_seed(1)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        torch.manual_seed(1)
        self.norm2 = torch.nn.BatchNorm2d(1, affine=False)
        torch.manual_seed(1)
        self.conv3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2)
        torch.manual_seed(1)
        self.norm3 = torch.nn.BatchNorm2d(1, affine=False)
    def forward(self, tensor3):
        x1 = self.conv1(tensor3)
        x2 = self.conv2(tensor3)
        x3 = self.conv3(tensor3)
        x = x1 + x2 + x3
        x = x - (x1 + x2 + x3) # output of conv3 is not used
        y = self.norm1(x1 + x2) # output of conv1 and conv2 are used together
        z = self.norm2(x3) # output of conv3 is used, and also other nodes,
        a = self.norm3(x) # but output of conv1, conv2 and conv3 are all used together
        y = 2 * y + z # y is used, but the output of z is not used
        x = x * (z + y) # x is not used
        return x
# Inputs to the model
tensor3 = torch.randn(1, 1, 64, 128)
