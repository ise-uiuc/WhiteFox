
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.153211984255, max_value=-0.913149786472):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose3d(5, 86, 3, stride=2, padding=0)
        self.conv1 = torch.nn.Conv3d(86, 53, 1, stride=1, padding=0)
        self.max_value = max_value
        self.min_value = min_value
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv1(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 5, 8, 80, 31)
