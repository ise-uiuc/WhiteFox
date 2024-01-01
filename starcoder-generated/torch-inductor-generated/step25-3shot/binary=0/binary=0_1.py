
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(288, 560, 1, stride=1, padding=1)
    def forward(self, x1, bias1=None, other=True):
        v1 = self.conv(x1)
        if bias1 == None:
            bias1 = torch.randn(v1.shape)
        else:
            v2 = v1 + bias1
            if other == True:
                v4 = v2 + torch.randn(v2.shape)
            return v4
# Inputs to the model
x1 = torch.randn(1, 288, 7, 7)
