
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(273, 77, 7, stride=1, bias=False)
    def forward(self, x):
        x45 = self.conv_t(x)
        x46 = x45 > 0
        x47 = x45 * 0.915
        x48 = torch.where(x46, x45, x47)
        return torch.flatten(x48, start_dim=1)
# Inputs to the model
x = torch.randn(2, 273, 88, 53)
