
class Model(torch.nn.Module):
    def __init__(self, min_value=-60.82, max_value=32):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 5, 1, stride=1, padding=1)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(5, 9, 1, stride=1, padding=1)
        self.conv_transpose3 = torch.nn.ConvTranspose2d(9, 13, 1, stride=1, padding=1)
        self.avg_pool2d = torch.nn.AvgPool2d(2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv_transpose2(v1)
        v3 = self.conv_transpose3(v2)
        v4 = self.avg_pool2d(v3)
        v5 = v4.view(v4.size(0), -1)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v8 = torch.reshape(v7, v4.shape)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 16, 16)
