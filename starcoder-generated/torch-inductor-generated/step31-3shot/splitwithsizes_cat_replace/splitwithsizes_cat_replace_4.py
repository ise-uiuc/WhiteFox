
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.ModuleList([torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=(1, 3), ), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 5, 1, 3), torch.nn.AdaptiveMaxPool2d(output_size=(7, 7)), torch.nn.Sigmoid(), torch.nn.Conv2d(64, 12, 1, 1, 0), torch.nn.Conv2d(12, 128, 3, 1, 1), torch.nn.GroupNorm(num_groups=5, num_channels=24, affine=True)])
    def forward(self, x, y):
        z1 = self.features[0](x)
        z1 = self.features[1](z1)
        z1 = self.features[2](z1)
        z1 = self.features[3](z1)
        z1 = self.features[4](z1)
        z1 = self.features[5](z1)
        z1 = self.features[6](z1)
        z1 = self.features[7](z1)
        z1 = self.features[8](z1)
        z2 = self.features[9](x)
        z2 = self.features[10](z2)
        o1 = y - z2
        o2 = o1 - z2[:, 3, ::].transpose(0, 1)
        return o2[:, :, :, :4]
# Inputs to the model
x = torch.randn(1, 3, 20, 20)
y = torch.randn(1, 20, 20, 16)
