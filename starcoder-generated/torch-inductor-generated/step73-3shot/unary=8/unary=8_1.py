
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(4, 4, 3, padding=2, output_padding=2, groups=6, bias=False)
        self.bn = torch.nn.BatchNorm2d(4)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool2d = torch.nn.MaxPool2d((3, 3), stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.bn(v1)
        v3 = self.relu(v2)
        v4 = self.max_pool2d(v3)
        v5 = v4 + 3
        v6 = torch.clamp(v5, min=0)
        v7 = torch.clamp(v6, max=6)
        v8 = v4 * v7
        v9 = v8 / 6
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 8, 8)
