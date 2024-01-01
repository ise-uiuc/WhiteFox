
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.modules.Sequential(conv_relu(3, 4), conv_relu(4, 8), conv_relu(8, 6), torch.nn.ReLU(inplace=True))
        self.transpose = torch.nn.Conv2d(6, 3, kernel_size=3, stride=2, padding=1, output_padding=1, groups=1, bias=False, dilation=1)
    def forward(self, x1):
        v1 = self.features(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64 )
