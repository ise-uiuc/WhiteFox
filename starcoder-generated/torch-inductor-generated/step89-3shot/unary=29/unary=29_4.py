
class Model(torch.nn.Module):
    def __init__(self, min_value=4.8168, max_value=5.6709):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(7, 4, 7, stride=2, padding=1, output_padding=1)
        self.conv_transpose3d = torch.nn.ConvTranspose3d(8, 4, (8, 8, 3), stride=[8, 1, 1], padding=0, output_padding=[5, 2, 2], groups=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose2d(x2)
        v3 = self.conv_transpose3d(v2)
        v4 = torch.clamp_min(v3, self.min_value)
        v5 = torch.clamp_max(v4, self.max_value)
        return v5
# Inputs to the model
x1 = torch.randn(1, 7, 51, 51)
x2 = torch.randn(1, 8, 125, 125, 24)
model = Model()
model(x1, x2)

# Model begins
class Model(torch.nn.Module):
    def __init__(self, min_value=1.0939, max_value=1.1002):
        super().__init__()
        self.max_pool2d = torch.nn.MaxPool2d((9, 1), stride=2, padding=(1, 2), dilation=(6, 1), ceil_mode=True)
        self.max_pool1d = torch.nn.MaxPool1d(2, stride=1, padding=2, dilation=1, ceil_mode=True)
        self.tanh = torch.nn.Tanh()
        self.tanh_ = torch.nn.Tanh()
        self.tanh_2 = torch.nn.Tanh()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1, x2):
        v1 = self.max_pool2d(x1)
        v2 = self.max_pool2d(x2)
        v3 = self.max_pool2d(v2)
        v4 = self.max_pool1d(v3)
        v5 = self.tanh_(v4)
        v6 = self.tanh(v5)
        v7 = self.tanh_2(v6)
        v8 = torch.clamp_min(v7, self.min_value)
        v9 = torch.clamp_max(v8, self.max_value)
        return v9
# Inputs to the model
x1 = torch.randn(1, 4, 101, 22)
x2 = torch.randn(1, 3, 32, 25)
model = Model()
model(x1, x2)
