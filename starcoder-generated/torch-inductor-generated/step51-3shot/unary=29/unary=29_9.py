
class Model(torch.nn.Module):
    def __init__(self, min_value=0.2, max_value=0.2):
        super().__init__()
        self.swish = torch.nn.SiLU()
        self.conv_transpose = torch.nn.ConvTranspose2d(2, 1, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.swish(v3)
        return v4
# Inputs to the model
x2 = torch.randn(1, 2, 96, 96)
