
class Model(torch.nn.Module):
    def __init__(self, min_value=4, max_value=-8):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x5):
        v1 = self.conv_transpose(x5)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.relu6(v3)
        return v4
# Inputs to the model
x5 = torch.randn(2, 1, 26, 26)
