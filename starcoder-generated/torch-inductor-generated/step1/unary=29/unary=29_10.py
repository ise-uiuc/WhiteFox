
class Model(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.convt = torch.nn.ConvTranspose2d(3, 8, 4, stride=2, **kwargs)

    def forward(self, x):
        return torch.clamp_max(torch.clamp_min(self.convt(x), min_value=-0.5), max_value=0.5)

# Initializing the model
m = Model(padding=1, output_padding=1)

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
