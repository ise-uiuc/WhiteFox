
class Model(torch.nn.Module):
    def __init__(self, min_value=2.1, max_value=3.9):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1, dilation=2)
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_relu(v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 3, 224, 224)
