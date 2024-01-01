
class Model(torch.nn.Module):
    def __init__(self, min_value=0, max_value=6.4):
        super(Model, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.avg_pool2d = torch.nn.AvgPool2d(2, stride=1, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        x = x.to(torch.float32)
        v1 = self.avg_pool2d(x)
        v6 = self.conv_transpose(v1)
        v2 = torch.clamp_min(v6, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_relu(v3)
        return v4
# Inputs to the model
x = torch.randn(1, 3, 224, 224)
