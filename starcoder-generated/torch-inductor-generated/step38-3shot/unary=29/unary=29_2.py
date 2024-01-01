
class Model(torch.nn.Module):
    def __init__(self, min_value=False, max_value=2.2):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.max_pool2d = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=2, return_indices=True, ceil_mode=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transpose3 = torch.nn.ConvTranspose3d(in_channels=2, out_channels=2, kernel_size=3, stride=(1, 1, 1), padding=(0, 1, 0), dilation=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(kernel_size=1, stride=1, padding=0, dilation=1)
    def forward(self, x4):
        v1 = self.conv_transpose2(x4)
        v2 = self.max_pool2d(v1)
        v3, v4 = v2[0], v2[1]
        v5 = v2[0].reshape(1, 2, 24, 6)
        v6 = self.conv_transpose3(v2)
        v7 = torch.clamp_min(v6, self.min_value)
        v8 = torch.clamp_max(v7, self.max_value)
        v9 = self.sigmoid(v8)
        return v9
# Inputs to the model
x4 = torch.randn(1, 2, 32, 30)
