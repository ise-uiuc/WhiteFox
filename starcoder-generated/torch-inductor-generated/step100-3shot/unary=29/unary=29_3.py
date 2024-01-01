
class Model(torch.nn.Module):
    def __init__(self, min_value=0.4417, max_value=0.8038):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 3, 5, stride=3, padding=2)
        self.conv = torch.nn.Conv2d(3, 3, 5, stride=3, padding=2)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(3, 3, 5, stride=2, padding=2)
        self.conv2 = torch.nn.Conv2d(3, 2, 5, stride=1, padding=2)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv(v3)
        v5 = self.conv_transpose2(v4)
        v6 = torch.clamp_min(v5, self.min_value)
        v7 = torch.clamp_max(v6, self.max_value)
        v8 = self.conv2(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 1, 59, 59)
