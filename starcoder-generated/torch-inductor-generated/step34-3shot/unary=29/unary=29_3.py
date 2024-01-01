
class Model(torch.nn.Module):
    def __init__(self, min_value=1.45927, max_value=4.771):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv_transpose = torch.nn.ConvTranspose3d(9, 3, kernel_size=3, stride=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu6(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 9, 67, 68, 81)
