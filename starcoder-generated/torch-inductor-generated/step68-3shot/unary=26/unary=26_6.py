
class Model(torch.nn.Module):
    def __init__(self, negative_slope):
        super().__init__()
        self.maxpool_with_argmax = torch.nn.MaxPool2d(kernel_size=10, stride=10, padding=0, dilation=1, return_indices=True)
        self.negative_slope = negative_slope
    def forward(self, x6):
        x7, x8 = self.maxpool_with_argmax(x6)
        x9 = x7 > 0
        x10 = x7 * self.negative_slope
        x11 = torch.where(x9, x7, x10)
        return x8 + torch.nn.functional.interpolate(x11, scale_factor=[1.0, 1.0])
negative_slope = 0.001
# Inputs to the model
x6 = torch.randn(78, 96, 1, 1)
