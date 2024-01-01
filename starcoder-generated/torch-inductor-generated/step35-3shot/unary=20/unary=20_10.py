
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.Sequential(torch.nn.Upsample(scale_factor=2), torch.nn.Sigmoid())
    def forward(self, x):
        return self.deconv(x)
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
