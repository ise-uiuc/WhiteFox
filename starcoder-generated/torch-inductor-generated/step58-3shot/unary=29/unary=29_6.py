
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(1, 48, 2, stride=4, padding=0)
        self.max_value = 1.173
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v3 = torch.clamp_max(v1, self.max_value)
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 1, 1)
