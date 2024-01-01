
class Model(torch.nn.Module):
    def __init__(self, min_value=1.5, max_value=3.8):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU()
        self.zero_pad2d = torch.nn.ZeroPad2d(1)
        self.constant_pad_nd = torch.nn.ConstantPad3d(2, 2)
        self.max_pool2d = torch.nn.MaxPool2d(3, stride=2, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.max_pool2d(x1)
        v2 = self.constant_pad_nd(v1)
        v3 = self.max_pool2d(v2)
        v4 = self.max_pool2d(v3)
        v5 = self.leaky_relu(v4)
        v6 = self.max_pool2d(v5)
        v7 = self.constant_pad_nd(v6)
        v8 = self.zero_pad2d(v7)
        v9 = torch.clamp_max(v8, self.max_value)
        return v9
# Inputs to the model
x1 = torch.randn(2, 6, 64, 64)
