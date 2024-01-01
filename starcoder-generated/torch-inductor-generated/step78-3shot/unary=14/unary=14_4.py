
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convtransposes = torch.nn.ModuleList([torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1), torch.nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)])
        self.modules = torch.nn.Sequential(*self.convtransposes)
    def forward(self, x):
        v = self.modules(x)
        v2 = torch.sigmoid(v)
        v3 = v * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 128, 128)
