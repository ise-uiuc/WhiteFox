
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x=None, padding=2, bias=None):
        if x == None:
            in_channels = 10
            x = torch.nn.functional.interpolate(torch.randn((1, in_channels, 64, 64)), size=(128, 128))
        v1 = self.conv(x)
        if bias == None:
            bias = torch.randn(v1.shape)
        v2 = v1 + bias
        if padding == 2:
            assert bias is None
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64) # or None to trigger random shape
