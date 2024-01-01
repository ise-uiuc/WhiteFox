
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The batch normalization layers include learnable parameter that needs
        # gradient update. Please initialize that parameter using either
        # `torch.nn.init.ones_(...)` or randomly sampled value. The dimension
        # depends on whether the input has batch size at the front
        if len(self.conv.weight.shape) == 4:
            out_channels = self.conv.weight.shape[0]
        else:
            out_channels = self.conv.weight.shape[-1]
        self.bn = torch.nn.BatchNorm2d(out_channels)

        self.conv = torch.nn.Conv2d(3, 64, 3, stride=1, padding=1)
        self.linear = torch.nn.Linear(64, 8)

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.bn(v1)
        v3 = v2.mean(dim=[2, 3])
        v4 = self.linear(v3)
        v5 = v4 ** 3
        return v5

# Initializing the model
m = Model()

# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
