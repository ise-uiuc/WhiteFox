
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, (1, 6), stride=1)
    def forward(self, x1):
        v1 = self.conv(x1) # Apply pointwise convolution with kernel size 1 to the input tensor
        v2 = v1.add(3) # Add 3 to the output of the convolution
        v3 = v2.clamp(min=0, max=6) # Clamp the output of the addition operation to a minimum of 0 and a maximum of 6
        v4 = v3.div(6) # Divide the output of the previous operation by 6
        return v4
# Inputs to the model
x1 = torch.randn(5, 3, 64, 64)
