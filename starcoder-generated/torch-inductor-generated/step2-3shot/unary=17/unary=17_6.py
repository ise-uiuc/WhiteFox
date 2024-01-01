
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.TConv1 = torch.nn.ConvTranspose2d(3, 8, 3, stride=2, padding=(1, 1))
        self.TConv2 = torch.nn.ConvTranspose2d(6, 8, 5, stride=1, padding=(2, 2), dilation=1)
    def forward(self, x1):
        v1 = self.TConv1(x1)
        v2 = self.TConv2(torch.cat((v1, x1),1))
        t2 = torch.relu(v2)
        return t2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
