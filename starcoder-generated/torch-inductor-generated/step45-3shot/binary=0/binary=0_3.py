
class Conv2dReLU(torch.nn.Sequential):
    def __init__(self, c_in, c_out, **kwargs):
        super().__init__(
            torch.nn.Conv2d(c_in, c_out, **kwargs),
            torch.nn.ReLU6(inplace=False)
        )


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 11, 3, stride=2, padding=1)
        self.model = torch.nn.Sequential(
            Conv2dReLU(2, 2, kernel_size=3, stride=2, padding=1),
            Conv2dReLU(2, 2, kernel_size=1, stride=1, padding=0),
            Conv2dReLU(2, 2, kernel_size=3, stride=2, padding=1),
        )
    def forward(self, x, other=0.1):
        x = self.conv(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = torch.flatten(x, 1)
        x = self.model(x)
        x = x + 1.1
        if other == False:
            other = torch.randn(v1.shape)
        x = x + other
        return x
# Inputs to the model
x1 = torch.randn(5, 3, 224, 224)
