
class Model(torch.nn.Module):
    def __init__(self, min_value=-0.3668, max_value=0.1731):
        super().__init__()
        self.conv_transpose2d = torch.nn.ConvTranspose2d(in_channels=3, out_channels=18, kernel_size=17, stride=3, padding=10)
        self.tanh = torch.nn.Tanh()
        self.conv_transpose3d = torch.nn.ConvTranspose3d(in_channels=5, out_channels=9, kernel_size=1, stride=1, padding=0)
        self.min_value = min_value
        self.max_value = max_value
    def forward(self, x2):
        v1 = self.conv_transpose2d(x2)
        v2 = torch.clamp_min(v1, self.min_value)
        v3 = torch.clamp_max(v2, self.max_value)
        return v3
# Inputs to the model
x2 = torch.randn(1, 3, 28, 28)
