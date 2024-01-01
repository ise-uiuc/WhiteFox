
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.04, max_value=-1.3):
        super().__init__()
        self.tanh = torch.nn.Tanh()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 8, 1, stride=1, padding=1)
        self.maxpool = torch.nn.MaxPool2d(2)
        self.padding_conv_transpose = torch.nn.ConvTranspose2d(8, 8, 2, stride=1, padding=9)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.tanh(v3)
        v5 = self.padding_conv_transpose(v4)
        v6 = self.maxpool(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
