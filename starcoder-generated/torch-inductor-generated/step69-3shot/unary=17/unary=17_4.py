
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 64, 1, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(2, 2)
        self.conv_transpose = torch.nn.ConvTranspose2d(64, 128, 3, stride=2, padding=1, dilation=2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(8)
        self.elu = torch.nn.ELU()
    def forward(self, x1):
        v1 = self.maxpool(self.relu(self.conv(x1)))
        _pad = torch.nn.ReflectionPad2d(2)
        v3 = self.maxpool(self.relu(self.conv(_pad(v1))))
        v5 = _pad(torch.relu(self.conv_transpose(v3)))
        v7 = torch.tanh(self.elu(self.avg_pool(v5)))
        return v7
# Inputs to the model
x1 = torch.randn(1, 2, 6, 6)
