
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.5, max_value=1):
        super().__init__()
        self.max_pool1d = torch.nn.MaxPool1d(5, stride=2, padding=1, dilation=3)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 1, stride=2, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.max_pool1d(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 5, 2)
