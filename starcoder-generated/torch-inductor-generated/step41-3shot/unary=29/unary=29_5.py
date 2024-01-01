
class Model(torch.nn.Module):
    def __init__(self, max_value=1):
        super().__init__()
        self.conv_transpose1 = torch.nn.ConvTranspose2d(3, 1, 4, stride=4, padding=4)
        self.conv_transpose2 = torch.nn.ConvTranspose2d(1, 1, 3, stride=3, padding=3)
        self.max_value = max_value
    def forward(self, x1):
        v1 = self.conv_transpose1(x1)
        v2 = torch.clamp_max(v1, self.max_value)
        v3 = self.conv_transpose2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
