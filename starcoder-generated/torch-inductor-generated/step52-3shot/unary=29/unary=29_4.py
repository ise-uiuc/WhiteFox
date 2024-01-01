
class Model(torch.nn.Module):
    def __init__(self, min_value=0.01, max_value=0.02):
        super().__init__()
        self.conv2d_2 = torch.nn.Conv2d(1, 2, 1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x11):
        v10 = self.conv2d_2(x11)
        v12 = torch.clamp_min(v10, self.min_value)
        v13 = torch.clamp_max(v12, self.max_value)
        return v13
# Inputs to the model
x11 = torch.randn(1, 1, 249, 249)
