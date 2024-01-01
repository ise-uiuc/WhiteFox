
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose3d(5, 16, 7, stride=6, groups=5, padding=8, dilation=2, bias=True)
        self.relu = torch.nn.ReLU()
        self.max_pool3d = torch.nn.MaxPool3d(kernel_size=10, stride=10, padding=8, dilation=6)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.relu(v1)
        v3 = self.max_pool3d(v2)
        v4 = v3 > 0
        v5 = v3 * 0.49
        v6 = torch.where(v4, v3, v5)
        return v6
# Inputs to the model
x3 = torch.randn(8, 5, 41, 32, 79)
