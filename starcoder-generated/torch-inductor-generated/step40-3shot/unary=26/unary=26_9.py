
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_t = torch.nn.ConvTranspose2d(81, 79, 3, stride=1, padding=0, bias=True)
    def forward(self, x12):
        l1 = self.conv_t(x12)
        s1 = torch.nn.functional.interpolate(l1, scale_factor=(0.226,), mode='nearest', align_corners=None)
        return s1
# Inputs to the model
x12 = torch.randn(52, 81, 52, 67)
