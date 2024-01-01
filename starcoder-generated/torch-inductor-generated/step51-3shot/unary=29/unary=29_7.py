
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.9, max_value=0.9):
        super().__init__()
        self.conv_transpose1d = torch.nn.ConvTranspose1d(2, 2, 3, stride=3)
        self.conv2d = torch.nn.Conv2d(2, 2, 3, stride=3)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x3):
        v1 = self.conv_transpose1d(x3)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        v4 = self.conv2d(v3)
        return v4
# Inputs to the model
x3 = torch.randn(1, 2, 40) 
