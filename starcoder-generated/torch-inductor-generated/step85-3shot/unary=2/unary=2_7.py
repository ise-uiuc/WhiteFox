
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(5, 6, (1, 2), stride=(1, 2)), 
                                          torch.nn.ConvTranspose2d(6, 5, (8, 8), stride=(2, 3)), 
                                          torch.nn.UpsamplingBilinear2d(scale_factor=3), torch.nn.ReLU6())
    def forward(self, x1):
        v1 = self.layers(x1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 5, 9, 1)
