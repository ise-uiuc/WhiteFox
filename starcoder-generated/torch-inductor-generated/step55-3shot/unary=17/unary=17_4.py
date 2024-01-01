
class Conv2dTranspose(torch.nn.ConvTranspose2d):
    def __init__(self, in_channels, *args, num_features=1, **kwargs):
        self.num_features = num_features
        super().__init__(in_channels, self.num_features, *args, **kwargs)

    def forward(self, x):
        x = super().forward(x)
        x = torch.nn.functional.upsample(x, size=(x.shape[-2], x.shape[-1]), mode="nearest")
        return x

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = Conv2dTranspose(3, 32, 3, padding=1, stride=2, num_features=32)
    def forward(self, x1):
        v1 = self.conv0(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
