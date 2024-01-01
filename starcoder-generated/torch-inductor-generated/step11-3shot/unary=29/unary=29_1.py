
class Model(torch.nn.Module):
    def __init__(self, min_value=0.5, max_value=0.7):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv_transposed = torch.nn.ConvTranspose2d(8, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.sigmoid(v3)
        v5 = self.conv_transposed(v4)
        return v1, v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
