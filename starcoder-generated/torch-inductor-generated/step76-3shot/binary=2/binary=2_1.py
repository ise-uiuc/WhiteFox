
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(out_channels=32, kernel_size=3, stride=2,
                                    padding=1, dilation=1, groups=1)
    def forward(self, _input):
        v1 = self.conv(_input)
        v2 = v1 - 1.0
        return v2
# Inputs to the model
input = torch.randn(128, 1, 128, 128)
