
class Model(torch.nn.Module):
    def __init__(self, min_value=-4, max_value=3):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 3, 4, stride=1, padding=0)
        self.softmax = torch.nn.Softmax2d()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 5, 3, stride=1, padding=0)
        self.avg_pool = torch.nn.AvgPool3d(kernel_size=3, stride=2, padding=1)
        self.dropout = torch.nn.Dropout3d()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v0 = self.conv(x1)
        v1 = self.softmax(v0)
        v2 = self.conv_transpose(v1)
        v3 = self.avg_pool(v2)
        v4 = self.dropout(v3)
        v5 = torch.clamp_min(v4, self.min_value)
        v6 = torch.clamp_max(v5, self.max_value)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 7, 7)
