
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1)
    def forward(self, x1, x2=None, dilations=3):
        v1 = self.conv(x1)
        if x2 == None:
            x2 = torch.randn(v1.shape)
        if dilations == 3:
            dilations = torch.randn(v1.shape).long()
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
