
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dconv = torch.nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, output_padding=1)
    def forward(self, x1):
        v1 = self.dconv(x1)
        v2 = torch.nn.functional.interpolate(v1, scale_factor=2.0, mode='bilinear') 
        v3 = torch.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 512, 4, 4)
