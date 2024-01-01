
class Model(torch.nn.Module):
    def __init__(self, min_value=-4.0, max_value=-1.0):
        super().__init__()
        self.conv = torch.nn.Conv3d(8, 4, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.sigmoid(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 8, 224, 224, 3)
