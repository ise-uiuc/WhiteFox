
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 10, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(10, 10, 1, stride=1, padding=1)
    def forward(self, x1):
        t1 = self.conv(x1)
        t2 = 3 + t1 # Add 3 to the output of the convolution
        t3 = torch.clamp(t2, 0, 6) # Clamp the output of the addition operation to a minimum of 0 and a maximum of 6
        t4 = t1 * t3 # Multiply the output of the convolution by the output of the clamp operation
        t5 = t4 / 6 # Divide the output of the multiplication operation by 6
        t6 = self.conv2(t5) # Apply pointwise convolution with kernel size 1 to the output of the division operation
        return t6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
