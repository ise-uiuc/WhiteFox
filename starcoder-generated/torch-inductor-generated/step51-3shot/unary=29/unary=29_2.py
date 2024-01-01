
class Model(torch.nn.Module):
    def __init__(self, min_value=4.99, max_value=4.0):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv2d = torch.nn.Conv2d(3, 3, 1, stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 6, 2, stride=2, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x4):
        v1 = self.conv2d(x4)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        v5 = self.relu6(v4)
        return v5
# Inputs to the model
x4 = torch.randn(1, 3, 8, 8)
