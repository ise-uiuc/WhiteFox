
class Model(torch.nn.Module):
    def __init__(self, min_value=-3.9641, max_value=-3.0173):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(10, 6, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x2 = torch.randn(1, 10, 3, 3)
