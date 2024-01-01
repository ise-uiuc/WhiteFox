
class Model(torch.nn.Module):
    def __init__(self, min_value=-3, max_value=3):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 1, 3, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.relu(x1)
        v2 = self.conv_transpose(v1)
        v3 = torch.clamp_min(v2, self.min_value)
        v4 = torch.clamp_max(v3, self.max_value)
        return v4
# Inputs to the model
x1 = torch.randn(4, 1, 5, 5)
