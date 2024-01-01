
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(43, 42, 1, stride=1, padding=0)
    def forward(self, x1):
        x2 = self.conv_t(x1)
        x3 = x2 > 0
        x4 = x2 * -0.608
        x5 = torch.where(x3, x2, x4)
        return torch.nn.functional.interpolate(x5, size=2, mode='bilinear', align_corners=None)    
# Inputs to the model
x1 = torch.randn((46, 43, 7, 15))
