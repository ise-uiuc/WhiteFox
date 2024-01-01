
class Model(torch.nn.Module):
    def __init__(self, min_value=0.2, max_value=2.5):
        super().__init__()
        self.conv_trans = torch.nn.ConvTranspose2d(3, 8, 1)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_trans(x1)
        v2 = v1.clamp(self.min_value, self.max_value)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
