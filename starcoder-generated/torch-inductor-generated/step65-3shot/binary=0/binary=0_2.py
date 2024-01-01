
class NFC(nn.Module):
    def __init__(self):
        super(NFC, self).__init__()

    def forward(self, x, y):
        A = F.relu(F.max_pool2d(self.conv1(x), 2)) # Apply pointwise convolution with kernel size 1 to the input tensor
        B = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(y)), 2))
# Inputs to the model
x1 = torch.randn(1, 4, 64, 64)
x2 = torch.randn(1, 16, 128, 128)
