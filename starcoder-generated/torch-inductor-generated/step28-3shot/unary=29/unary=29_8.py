
class Model(torch.nn.Module):
    def __init__(self, min_value=17103715, max_value=184578103):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 64, kernel_size=(2, 2), stride=(2, 2), padding=(1, 1))
        self.conv2d = torch.nn.Conv2d(64, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.conv_transpose3d = torch.nn.ConvTranspose3d(256, 3, kernel_size=(4, 3, 2), stride=(3, 1, 5), padding=(1, 0, 5))
        self.tanh = torch.nn.Tanh()
        self.max_pool3d = torch.nn.MaxPool3d(kernel_size=2, stride=1, padding=1)
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        self.avg_pool2d = torch.nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        v1 = torch.clamp_min(self.conv_transpose2d(x), self.min_value)
        v2 = torch.clamp_max(v1, self.max_value)
        v7 = self.conv2d(v2)
        v6 = torch.clamp_min(v7, self.min_value)
        v8 = torch.clamp_max(v6, self.max_value)
        v9 = self.conv_transpose3d(v8)
        v11 = self.tanh(v9)
        v3 = torch.clamp_min(v11, self.min_value)
        v4 = torch.clamp_max(v3, self.min_value)
        v10 = self.max_pool3d(v4)
        v5 = torch.clamp_min(v10, self.max_value)
        v12 = self.max_pool2d(v5)
        v13 = self.tanh(v12)
        v14 = self.avg_pool2d(v13)
        return v14
# Inputs to the model
x1 = torch.randn(1, 3, 108, 7, 7)
