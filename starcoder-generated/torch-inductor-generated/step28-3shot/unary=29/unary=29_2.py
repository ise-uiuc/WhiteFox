
class Model(torch.nn.Module):
    def __init__(self, min_value=-1000.9, max_value=-41.3):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=3)
        self.conv = torch.nn.Conv2d(1, 2, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 4, 1, stride=1)
        self.act = torch.nn.ReLU6()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.act(v4)
        v6 = self.max_pool2d(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 1, 128, 128)
