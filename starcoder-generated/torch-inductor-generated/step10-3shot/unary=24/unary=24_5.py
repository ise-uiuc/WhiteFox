
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 32, 1, stride=3, padding=1, dilation=4)
        self.negative_slope = negative_slope
        self.pool = torch.nn.AvgPool2d(2, 3, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.pool(v1)
        v3 = v2 > 0 # Create a boolean mask where each element is True if the corresponding element in v2 is greater than 0, False otherwise
        v4 = v2 * self.negative_slope # Multiply the output of the pooling by the negative_slope if the mask element is True, otherwise output the pooling result element
        v5 = torch.where(v3, v2, v4)
        return v5
negative_slope = 1
# Inputs to the model
x1 = torch.randn(1, 8, 64, 64)
