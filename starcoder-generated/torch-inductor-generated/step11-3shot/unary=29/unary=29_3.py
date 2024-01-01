
class Model(torch.nn.Module):
    def __init__(self, min_value=-1.2, max_value=0.4):
        super(Model, self).__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        x2 = self.conv_transpose(x1)
        x3 = torch.clamp_min(x2, self.min_value)
        x4 = torch.clamp_max(x3, self.max_value)
        return x4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
