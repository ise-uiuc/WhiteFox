
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(16, 32, 6, stride=6, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.rsqrt(v1)
        v3 = v2 * 0.3333333333333333 # Multiply the output of the sqrt with 0.3333333333333333
        v4 = torch.sigmoid(v3) # Apply the sigmoid to the output of the sqrt
        v5 = v2 * v4 # Multiply the output of the sqrt by the output of the sigmoid
        return v5
# Inputs to the model
x1 = torch.randn(1, 16, 21, 21)
