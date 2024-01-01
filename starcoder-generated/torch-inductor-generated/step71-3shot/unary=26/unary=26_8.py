
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 57, 4, stride=1, padding=0, bias=True)
    def forward(self, x1):
        o1 = self.conv_t(x1)
        o2 = o1 > 0
        o3 = o1 * 0.0377
        o4 = torch.where(o2, o1, o3)
        return o4
# Inputs to the model
x1 = torch.randn(5488822, 1, 1, 1)
