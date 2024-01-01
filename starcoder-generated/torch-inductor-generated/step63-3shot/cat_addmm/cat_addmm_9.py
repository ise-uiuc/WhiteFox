
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Upsample(scale_factor=2, mode='linear')
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.layers(x)
        x = x.permute(0, 3, 1, 2)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.flatten(x, start_dim=3)
        x = torch.cat((x, x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 1, 2, 4)
