
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.ConvTranspose2d(3, 8, 3, stride=1, padding=1, output_padding=0, groups=1, bias=True)
        self.min_value = 0
        self.max_value = 0.5
    
    def forward(self, x):
        v1 = self.deconv(x)
        v2 = torch.clamp_max(torch.clamp(v1, min=self.min_value), self.max_value)
        return v2

# Initializing the model
m = Model()

# Inputs to the model
x = torch.randn(1, 3, 64, 64)
