
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        torch.nn.init.zeros_(self.conv.weight) # Set the weight tensor such that the result of the pointwise convolution is all zeros.

    def forward(self, x):
        v1 = self.conv(x) # Run the convolution operator
        return torch.sigmoid(v1) # Apply sigmoid operator to the previous result
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
