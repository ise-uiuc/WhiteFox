
class Model(torch.nn.Module):
    def __init__(self, min_value=73, max_value=82):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 2, 2, stride=2, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x):
        v1 = self.conv_transpose(x)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(3, 2, 4, 5)
