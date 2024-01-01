
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.7, max_value=3):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.clamp = torch.nn.Clamp(min=min_value, max=max_value)
        self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_transpose2d(x1)
        v2 = self.clamp(v1)
        v3 = self.sigmoid(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
