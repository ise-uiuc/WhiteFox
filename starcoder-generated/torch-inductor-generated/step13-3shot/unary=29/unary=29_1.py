
class Model(torch.nn.Module):
    def __init__(self, min_value=-15, max_value=-8):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 6, 3, padding=(0, 1), stride=1)
        self.conv_transpose = torch.nn.ConvTranspose2d(6, 2, 3, padding=(1, 0), stride=1)
        self.clamp_min = torch.nn.Hardtanh(min_value=min_value, max_value=max_value, inplace=False)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.conv_transpose(v1)
        v3 = self.clamp_min(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
