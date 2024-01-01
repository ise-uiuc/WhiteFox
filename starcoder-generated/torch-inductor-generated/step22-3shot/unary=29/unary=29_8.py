
class Model(torch.nn.Module):
    def __init__(self, min_value, max_value):
        super(Model, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.max_pool2d = torch.nn.MaxPool2d(8, stride=2, padding=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 32, 8, stride=2, padding=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv_transpose(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.leaky_relu(v3)
        v5 = self.max_pool2d(v4)
        return v5
min_value = -0.5
max_value = 0.5
# Inputs to the model
x3 = torch.randn(1, 3, 224, 256)
