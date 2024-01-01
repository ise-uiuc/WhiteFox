
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU6(inplace=True)
        self.negative_slope = negative_slope
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = self.relu(x2)
        x4 = x3 * self.negative_slope
        x5 = torch.where(x3, x2, x4)
        return x5
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
